//! Local module-graph traversal with no compilation or macro expansion.
//!
//! Canonical paths establish containment before each read. The active recursion
//! stack detects cycles independently of repeated aliases. All aliases retain
//! their own classification contexts; first-visit deduplication must not erase
//! a production reference to a file first encountered through a test module.

use std::{
    collections::{BTreeMap, BTreeSet},
    path::{Path, PathBuf},
};

use syn::{
    Attribute, Item,
    spanned::Spanned,
    visit::{self, Visit},
};

use crate::{
    cfg::{self, Reach},
    source::*,
};

/// Maximum source buffers/references accepted by one bounded audit invocation.
const MAX_CONTEXTS: usize = 4096;
/// Maximum nested module/item traversal depth, independent of source length.
const MAX_DEPTH: usize = 128;
/// Maximum bytes read from one source file before parsing it.
const MAX_FILE_BYTES: u64 = 8 * 1024 * 1024;

/// Classifies explicit crate entry files beneath one canonical source root.
///
/// Entries may be absolute or root-relative but must canonicalize beneath the
/// root. Source errors become visible diagnostics and retained source whenever
/// a readable UTF-8 buffer exists. Invalid roots, entry escapes, non-UTF-8
/// source, and an empty entry list return an error. No Cargo graph is inferred.
pub fn classify(root: &Path, entries: &[PathBuf]) -> Result<Report, String> {
    let root = root
        .canonicalize()
        .map_err(|error| format!("canonicalize root: {error}"))?;
    if !root.is_dir() || entries.is_empty() {
        return Err("a directory root and at least one entry are required".into());
    }
    let mut scan = Scan {
        root,
        snapshots: BTreeMap::new(),
        seen: BTreeSet::new(),
        active: Vec::new(),
        diagnostics: Vec::new(),
        contexts: 0,
    };
    let mut entry_names = Vec::new();
    for (index, entry) in entries.iter().enumerate() {
        let path = scan
            .root
            .join(entry)
            .canonicalize()
            .map_err(|error| format!("resolve entry {}: {error}", entry.display()))?;
        if !path.starts_with(&scan.root) || !path.is_file() {
            return Err(format!(
                "entry {} is not a file contained by root",
                entry.display()
            ));
        }
        entry_names.push(scan.relative(&path));
        let directory = path.parent().ok_or("entry has no parent")?.to_owned();
        scan.file(
            &path,
            format!("entry[{index}]"),
            directory,
            Reach::root(),
            true,
            None,
        )?;
    }
    // A diagnostic describing unresolved source is an explicit retention
    // override, even if an enclosing cfg would otherwise prove test-only.
    for issue in &scan.diagnostics {
        if (issue.code.starts_with("unresolved")
            || issue.code.starts_with("ambiguous")
            || issue.code == "module_cycle"
            || issue.code == "traversal_limit")
            && let Some(snapshot) = scan.snapshots.get_mut(&scan.root.join(&issue.file))
        {
            let span = issue
                .span
                .clone()
                .unwrap_or_else(|| snapshot.source.whole_span());
            for context in &mut snapshot.contexts {
                context.regions.push(Region {
                    span: span.clone(),
                    depth: usize::MAX,
                    kind: "unresolved_retention_override".into(),
                    test_only: false,
                    reason: issue.reason.clone(),
                });
            }
        }
    }
    let files = scan
        .snapshots
        .into_iter()
        .map(|(path, snapshot)| {
            let lines = snapshot.source.classify_lines(&snapshot.contexts);
            let relative = path
                .strip_prefix(&scan.root)
                .unwrap_or(&path)
                .to_string_lossy()
                .into_owned();
            for run in &lines {
                if run.classification == Classification::Mixed {
                    scan.diagnostics.push(Diagnostic {
                        code: "mixed_production_test_lines".into(),
                        file: relative.clone(),
                        span: None,
                        reason: format!(
                            "lines {}..={} have both retained and test-only ownership; retained",
                            run.start_line, run.end_line
                        ),
                    });
                }
            }
            FileReport {
                path: relative,
                sha256: snapshot.source.sha256(),
                byte_length: snapshot.source.text.len(),
                line_count: snapshot.source.line_count(),
                contexts: snapshot.contexts,
                lines,
                test_audit: snapshot.test_audit,
            }
        })
        .collect();
    Ok(Report {
        schema_version: 1,
        purpose: "supplementary_source_classification_not_a_coverage_gate",
        qualified_for_coverage_exclusions: false,
        macro_expansion_resolved: false,
        root: scan.root.to_string_lossy().into_owned(),
        entries: entry_names,
        files,
        diagnostics: scan.diagnostics,
    })
}

/// A physical source snapshot shared by every module-reference context.
struct Snapshot {
    /// Immutable original source buffer and coordinate index.
    source: Source,
    /// Every visited context, including repeated aliases.
    contexts: Vec<Context>,
    /// Independent parsed-attribute audit of these exact physical source bytes.
    test_audit: crate::TestAudit,
}

/// Bounded traversal state shared across independently supplied entry roots.
struct Scan {
    /// Canonical containment boundary for source reads.
    root: PathBuf,
    /// One exact buffer per canonical physical source path.
    snapshots: BTreeMap<PathBuf, Snapshot>,
    /// Equivalent file/directory/reachability contexts already encountered.
    seen: BTreeSet<String>,
    /// Active physical paths for recursion-cycle detection.
    active: Vec<PathBuf>,
    /// Visible unresolved/ambiguous syntax and graph observations.
    diagnostics: Vec<Diagnostic>,
    /// Count of attempted module-reference contexts.
    contexts: usize,
}

impl Scan {
    /// Displays a canonical path relative to the source boundary.
    fn relative(&self, path: &Path) -> String {
        path.strip_prefix(&self.root)
            .unwrap_or(path)
            .to_string_lossy()
            .into_owned()
    }

    /// Records a contextual diagnostic without silently excluding its source.
    fn issue(
        &mut self,
        path: &Path,
        span: Option<SourceSpan>,
        code: &str,
        reason: impl Into<String>,
    ) {
        self.diagnostics.push(Diagnostic {
            code: code.into(),
            file: self.relative(path),
            span,
            reason: reason.into(),
        });
    }

    /// Reads or reuses one source and visits it in a distinct module context.
    fn file(
        &mut self,
        requested: &Path,
        module: String,
        directory: PathBuf,
        mut reach: Reach,
        entry: bool,
        origin: Option<(&Path, SourceSpan)>,
    ) -> Result<(), String> {
        let path = match requested.canonicalize() {
            Ok(path) if path.starts_with(&self.root) && path.is_file() => path,
            result => {
                let reason = match result {
                    Ok(_) => format!(
                        "module {} escapes root or is not a regular file",
                        requested.display()
                    ),
                    Err(error) => format!("cannot resolve module {}: {error}", requested.display()),
                };
                let (owner, span) =
                    origin.map_or((requested, None), |(path, span)| (path, Some(span)));
                self.issue(owner, span, "unresolved_module", reason);
                return Ok(());
            }
        };
        self.contexts += 1;
        if self.contexts > MAX_CONTEXTS || self.active.len() >= MAX_DEPTH {
            let (owner, span) =
                origin.map_or((path.as_path(), None), |(path, span)| (path, Some(span)));
            self.issue(
                owner,
                span,
                "traversal_limit",
                "module traversal limit reached; descendants unresolved",
            );
            return Ok(());
        }
        if self.active.contains(&path) {
            let (owner, span) =
                origin.map_or((path.as_path(), None), |(path, span)| (path, Some(span)));
            self.issue(
                owner,
                span,
                "module_cycle",
                format!("active recursion cycle at {module}; descendants unresolved"),
            );
            return Ok(());
        }
        if !self.snapshots.contains_key(&path) {
            let file = std::fs::File::open(&path)
                .map_err(|error| format!("open {}: {error}", path.display()))?;
            use std::io::Read;
            let mut bytes = Vec::new();
            file.take(MAX_FILE_BYTES + 1)
                .read_to_end(&mut bytes)
                .map_err(|error| format!("read {}: {error}", path.display()))?;
            if bytes.len() as u64 > MAX_FILE_BYTES {
                return Err(format!(
                    "source {} exceeds {MAX_FILE_BYTES} bytes",
                    path.display()
                ));
            }
            let text = String::from_utf8(bytes)
                .map_err(|error| format!("source {} is not UTF-8: {error}", path.display()))?;
            self.snapshots.insert(
                path.clone(),
                Snapshot {
                    source: Source::new(text),
                    contexts: Vec::new(),
                    test_audit: crate::test_audit::unparsed(),
                },
            );
        }
        let text = &self.snapshots[&path].source.text;
        let parsed = syn::parse_file(text);
        let bom = usize::from(text.starts_with('\u{feff}')) * '\u{feff}'.len_utf8();
        let mut offset = bom;
        if let Ok(file) = &parsed {
            offset += file.shebang.as_ref().map_or(0, String::len);
            let (own, issues) = cfg::attributes(&file.attrs);
            reach = reach.and(own);
            for issue in issues {
                self.issue(&path, None, "unresolved_cfg", issue);
            }
        } else {
            reach.unresolved = true;
        }
        let test_audit =
            crate::test_audit::audit(&self.snapshots[&path].source, parsed.as_ref().ok(), offset);
        self.snapshots
            .get_mut(&path)
            .ok_or("source snapshot disappeared")?
            .test_audit = test_audit;
        let key = format!("{}|{}|{reach:?}", path.display(), directory.display());
        let duplicate_context = !self.seen.insert(key);
        if duplicate_context {
            self.issue(
                &path,
                None,
                "duplicate_context",
                "equivalent source context revisited; ownership remains represented",
            );
        }
        let root_region = Region {
            span: self.snapshots[&path].source.whole_span(),
            depth: 0,
            kind: "file".into(),
            test_only: reach.test_only(),
            reason: reach.reason().into(),
        };
        let mut context = Context {
            module: module.clone(),
            duplicate_context,
            regions: vec![root_region],
        };
        match parsed {
            Ok(file) => {
                self.active.push(path.clone());
                let default_directory =
                    if entry || path.file_name().is_some_and(|name| name == "mod.rs") {
                        path.parent().unwrap_or(&directory).to_owned()
                    } else {
                        path.with_extension("")
                    };
                let mut visitor = Walker {
                    scan: self,
                    path: &path,
                    context: &mut context,
                    reach,
                    depth: 0,
                    offset,
                    module,
                    directory: default_directory,
                    inline_depth: 0,
                    failure: None,
                };
                for item in &file.items {
                    visitor.visit_item(item);
                }
                let failure = visitor.failure.take();
                self.active.pop();
                if let Some(error) = failure {
                    return Err(error);
                }
            }
            Err(error) => self.issue(
                &path,
                None,
                "parse_error",
                format!("source retained without syntax classification: {error}"),
            ),
        }
        self.snapshots
            .get_mut(&path)
            .ok_or("source snapshot disappeared")?
            .contexts
            .push(context);
        Ok(())
    }
}

/// Per-file AST visitor; inherited reachability cannot be widened by children.
struct Walker<'a> {
    /// Shared module graph and diagnostics.
    scan: &'a mut Scan,
    /// Canonical physical file currently parsed.
    path: &'a Path,
    /// Regions accumulated for this reference, not other aliases.
    context: &'a mut Context,
    /// Conjunction of all enclosing cfg constraints.
    reach: Reach,
    /// Current AST ownership nesting level.
    depth: usize,
    /// BOM/shebang byte prefix removed by Syn's parse_file.
    offset: usize,
    /// Logical module path used for provenance.
    module: String,
    /// Default external-module search directory for this nesting level.
    directory: PathBuf,
    /// Inline-module depth controls explicit path-attribute resolution.
    inline_depth: usize,
    /// Fatal source IO/encoding failure deferred through the visitor interface.
    failure: Option<String>,
}

impl Walker<'_> {
    /// Converts a parser span back to exact original-source byte coordinates.
    fn span(&self, span: proc_macro2::Span) -> Option<SourceSpan> {
        let range = span.byte_range();
        self.scan.snapshots[self.path].source.span(
            range.start.checked_add(self.offset)?,
            range.end.checked_add(self.offset)?,
        )
    }

    /// Pushes one item ownership region and conjoins its configuration.
    fn enter(&mut self, attrs: &[Attribute], span: proc_macro2::Span, kind: &str) -> Reach {
        let previous = self.reach;
        let (own, issues) = cfg::attributes(attrs);
        self.reach = self.reach.and(own);
        self.depth += 1;
        let mapped = self.span(span);
        for issue in issues {
            self.scan
                .issue(self.path, mapped.clone(), "unresolved_cfg", issue);
        }
        if let Some(span) = mapped {
            self.context.regions.push(Region {
                span,
                depth: self.depth,
                kind: kind.into(),
                test_only: self.reach.test_only(),
                reason: self.reach.reason().into(),
            });
        } else {
            self.context.regions.push(Region {
                span: self.scan.snapshots[self.path].source.whole_span(),
                depth: usize::MAX,
                kind: "invalid_span_fallback".into(),
                test_only: false,
                reason: "parser span did not map to original bytes; retained".into(),
            });
            self.scan.issue(
                self.path,
                None,
                "unresolved_span",
                "invalid parser source coordinates; entire file retained",
            );
        }
        previous
    }

    /// Restores parent reachability after a complete item's descendants.
    fn leave(&mut self, previous: Reach) {
        self.reach = previous;
        self.depth -= 1;
    }

    /// Resolves explicit module paths according to inline/non-inline rules.
    fn path_base(&self) -> PathBuf {
        if self.inline_depth == 0 {
            self.path.parent().unwrap_or(&self.directory).to_owned()
        } else {
            self.directory.clone()
        }
    }

    /// Walks inline modules or every conservative external path candidate.
    fn module(&mut self, item: &syn::ItemMod) {
        let (paths, conditional) = cfg::paths(&item.attrs);
        let ambiguous = conditional || paths.len() > 1;
        if ambiguous {
            self.scan.issue(self.path, self.span(item.span()), "ambiguous_module_path", "conditional, malformed, or multiple path attributes; every known candidate retained and unresolved");
        }
        let mut next_reach = self.reach;
        next_reach.unresolved |= ambiguous;
        let name = item.ident.to_string().trim_start_matches("r#").to_owned();
        let logical = format!("{}::{name}", self.module);
        if let Some((_, items)) = &item.content {
            let old_directory = self.directory.clone();
            let old_module = self.module.clone();
            let old_reach = self.reach;
            let mut candidates: Vec<_> = paths
                .iter()
                .map(|path| self.path_base().join(path))
                .collect();
            if paths.is_empty() || ambiguous {
                candidates.push(self.directory.join(&name));
            }
            candidates.sort();
            candidates.dedup();
            for directory in candidates {
                self.directory = directory;
                self.module = logical.clone();
                self.reach = next_reach;
                self.inline_depth += 1;
                for child in items {
                    self.visit_item(child);
                }
                self.inline_depth -= 1;
            }
            self.directory = old_directory;
            self.module = old_module;
            self.reach = old_reach;
            return;
        }
        let mut candidates: Vec<PathBuf> = paths
            .iter()
            .map(|path| self.path_base().join(path))
            .collect();
        if paths.is_empty() || ambiguous {
            let defaults = [
                self.directory.join(format!("{name}.rs")),
                self.directory.join(&name).join("mod.rs"),
            ];
            let existing: Vec<_> = defaults
                .iter()
                .filter(|path| path.exists())
                .cloned()
                .collect();
            if existing.len() == 2 {
                next_reach.unresolved = true;
                self.scan.issue(
                    self.path,
                    self.span(item.span()),
                    "ambiguous_module_files",
                    "both name.rs and name/mod.rs exist; both retained as unresolved",
                );
            }
            if existing.is_empty() {
                self.scan.issue(
                    self.path,
                    self.span(item.span()),
                    "unresolved_module",
                    format!("no default source for {logical}"),
                );
            }
            candidates.extend(existing);
        }
        candidates.sort();
        candidates.dedup();
        for candidate in candidates {
            if let Some(span) = self.span(item.span()) {
                let directory = candidate.parent().unwrap_or(&self.directory).to_owned();
                if let Err(error) = self.scan.file(
                    &candidate,
                    logical.clone(),
                    directory,
                    next_reach,
                    false,
                    Some((self.path, span)),
                ) {
                    self.failure = Some(error);
                }
            }
        }
    }
}

impl<'ast> Visit<'ast> for Walker<'_> {
    /// Visits actual parsed items, never fake syntax in strings or comments.
    fn visit_item(&mut self, item: &'ast Item) {
        if self.depth >= MAX_DEPTH || self.failure.is_some() {
            self.scan.issue(
                self.path,
                self.span(item.span()),
                "traversal_limit",
                "item nesting limit or earlier fatal error; descendants unresolved",
            );
            return;
        }
        let (attrs, kind) = item_attributes(item);
        let previous = self.enter(attrs, item.span(), kind);
        match item {
            Item::Mod(module) => self.module(module),
            Item::Macro(_) | Item::Verbatim(_) => {
                self.scan.issue(self.path, self.span(item.span()), "opaque_macro_or_verbatim", "macro/generated syntax is not expanded or excluded without inherited cfg proof");
            }
            _ => visit::visit_item(self, item),
        }
        self.leave(previous);
    }

    /// Tracks cfg ownership on associated implementation items.
    fn visit_impl_item(&mut self, item: &'ast syn::ImplItem) {
        let attrs: &[Attribute] = match item {
            syn::ImplItem::Const(value) => &value.attrs,
            syn::ImplItem::Fn(value) => &value.attrs,
            syn::ImplItem::Type(value) => &value.attrs,
            syn::ImplItem::Macro(value) => &value.attrs,
            _ => &[],
        };
        let previous = self.enter(attrs, item.span(), "impl_item");
        visit::visit_impl_item(self, item);
        self.leave(previous);
    }

    /// Tracks cfg ownership on associated trait items.
    fn visit_trait_item(&mut self, item: &'ast syn::TraitItem) {
        let attrs: &[Attribute] = match item {
            syn::TraitItem::Const(value) => &value.attrs,
            syn::TraitItem::Fn(value) => &value.attrs,
            syn::TraitItem::Type(value) => &value.attrs,
            syn::TraitItem::Macro(value) => &value.attrs,
            _ => &[],
        };
        let previous = self.enter(attrs, item.span(), "trait_item");
        visit::visit_trait_item(self, item);
        self.leave(previous);
    }
}

/// Retrieves attributes from each known Syn item without textual heuristics.
fn item_attributes(item: &Item) -> (&[Attribute], &'static str) {
    match item {
        Item::Const(value) => (&value.attrs, "const"),
        Item::Enum(value) => (&value.attrs, "enum"),
        Item::ExternCrate(value) => (&value.attrs, "extern_crate"),
        Item::Fn(value) => (&value.attrs, "fn"),
        Item::ForeignMod(value) => (&value.attrs, "foreign_mod"),
        Item::Impl(value) => (&value.attrs, "impl"),
        Item::Macro(value) => (&value.attrs, "macro"),
        Item::Mod(value) => (&value.attrs, "mod"),
        Item::Static(value) => (&value.attrs, "static"),
        Item::Struct(value) => (&value.attrs, "struct"),
        Item::Trait(value) => (&value.attrs, "trait"),
        Item::TraitAlias(value) => (&value.attrs, "trait_alias"),
        Item::Type(value) => (&value.attrs, "type"),
        Item::Union(value) => (&value.attrs, "union"),
        Item::Use(value) => (&value.attrs, "use"),
        _ => (&[], "unresolved_item"),
    }
}
