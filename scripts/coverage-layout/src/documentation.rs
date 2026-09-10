//! Source-only enforcement of authored function and module documentation style.
//!
//! Syn supplies item and attribute boundaries; the existing classifier supplies
//! all module-reference contexts and structural test ownership. Literal `///`
//! and `//!` are distinguished from equivalent rustdoc attributes. Macro item
//! fragments are inspected when they parse; unresolved authored function tokens
//! remain explicit review blockers, never silently generated or documented code.

use std::{
    collections::BTreeSet,
    path::{Path, PathBuf},
};

use proc_macro2::{TokenStream, TokenTree};
use serde::Serialize;
use syn::{
    AttrStyle, Attribute, Meta,
    spanned::Spanned,
    visit::{self, Visit},
};

use crate::{FileReport, SourceSpan, source::Source};

/// One versioned audit of the explicit entry graph's exact source snapshots.
#[derive(Debug, Serialize)]
pub struct DocumentationReport {
    /// Independent schema for documentation findings and macro review records.
    pub schema_version: u32,
    /// Canonical source containment root.
    pub root: String,
    /// Crate entry points supplied by the caller.
    pub entries: Vec<String>,
    /// Always false: derive/procedural/declarative expansion is outside this audit.
    pub macro_expansion_resolved: bool,
    /// Whether all authored syntax was resolved without opaque function tokens.
    pub complete: bool,
    /// True only when complete and every production site uses its required style.
    pub passed: bool,
    /// Audited physical sources in deterministic path order.
    pub files: Vec<DocumentationFile>,
    /// Parse, graph, source-change, or macro-template limitations requiring action.
    pub diagnostics: Vec<crate::Diagnostic>,
}

/// One authenticated file with parsed sites and unresolved macro boundaries.
#[derive(Debug, Serialize)]
pub struct DocumentationFile {
    /// Root-relative canonical source path.
    pub path: String,
    /// SHA-256 of the exact source bytes that were parsed.
    pub sha256: String,
    /// Number of bytes in that source snapshot.
    pub byte_length: usize,
    /// Authored functions/methods and physical or inline module definitions.
    pub sites: Vec<DocumentationSite>,
    /// Macro boundaries, including successfully parsed fragments and opaque tokens.
    pub macros: Vec<MacroBoundary>,
    /// Inline test-module bodies that must move to a sibling test source.
    pub inline_test_bodies: Vec<SourceSpan>,
}

/// One real syntax node; visibility never exempts a production helper.
#[derive(Debug, Serialize)]
pub struct DocumentationSite {
    /// Source span including attributes and function/module body.
    pub span: SourceSpan,
    /// free_function, impl_method, trait_method, foreign_function, or module kind.
    pub kind: String,
    /// Authored name, or the physical filename for a file module.
    pub name: String,
    /// parsed_item or parseable_macro_fragment, never compiler-expanded syntax.
    pub origin: String,
    /// Whether all observed contexts and structural attributes prove test ownership.
    pub test_only: bool,
    /// Required literal spelling: /// for functions or //! for module definitions.
    pub required_style: &'static str,
    /// Nonempty correctly placed literal line documentation satisfies the style.
    pub style_present: bool,
    /// Equivalent literal #[doc] or block docs are reported rather than called absent.
    pub rustdoc_present: bool,
    /// A computed or conditional doc attribute cannot establish the literal style.
    pub computed_documentation: bool,
}

/// A macro token boundary whose expansion this source audit does not invent.
#[derive(Debug, Serialize)]
pub struct MacroBoundary {
    /// Exact enclosing invocation or macro_rules definition span.
    pub span: SourceSpan,
    /// Macro path or macro_rules definition name.
    pub name: String,
    /// Actual test-owned macros require no production documentation review.
    pub test_only: bool,
    /// Unresolved authored `fn name` / `fn $name` tokens, excluding function-pointer types.
    pub unresolved_functions: Vec<SourceSpan>,
}

/// Audits the parsed module graph while preserving its source and cfg evidence.
pub fn audit_documentation(
    root: &Path,
    entries: &[PathBuf],
) -> Result<DocumentationReport, String> {
    let layout = crate::classify(root, entries)?;
    let root = Path::new(&layout.root);
    let mut diagnostics: Vec<_> = layout
        .diagnostics
        .into_iter()
        .filter(|issue| {
            !matches!(
                issue.code.as_str(),
                "opaque_macro_or_verbatim" | "mixed_production_test_lines"
            )
        })
        .collect();
    let mut files = Vec::new();
    for file in &layout.files {
        let text = std::fs::read_to_string(root.join(&file.path))
            .map_err(|error| format!("read documentation source {}: {error}", file.path))?;
        let source = Source::new(text);
        if source.sha256() != file.sha256 {
            return Err(format!(
                "{} changed between classification and documentation parsing",
                file.path
            ));
        }
        let parsed = match syn::parse_file(&source.text) {
            Ok(parsed) => parsed,
            Err(_) => continue, // The classifier's parse diagnostic remains a blocker.
        };
        let offset = usize::from(source.text.starts_with('\u{feff}')) * '\u{feff}'.len_utf8()
            + parsed.shebang.as_ref().map_or(0, String::len);
        let mut walker = DocumentationWalker {
            source: &source,
            file,
            offset,
            sites: Vec::new(),
            macros: Vec::new(),
            seen: BTreeSet::new(),
            origin: "parsed_item",
            depth: 0,
            diagnostics: &mut diagnostics,
            local_test_spans: Vec::new(),
            inline_test_bodies: Vec::new(),
        };
        walker.record_module(source.whole_span(), &file.path, &parsed.attrs);
        walker.visit_file(&parsed);
        walker
            .sites
            .sort_by_key(|site| (site.span.start.byte, site.span.end.byte));
        files.push(DocumentationFile {
            path: file.path.clone(),
            sha256: file.sha256.clone(),
            byte_length: file.byte_length,
            sites: walker.sites,
            macros: walker.macros,
            inline_test_bodies: walker.inline_test_bodies,
        });
    }
    let complete = diagnostics
        .iter()
        .all(|issue| issue.code == "duplicate_context")
        && files.iter().all(|file| {
            file.macros
                .iter()
                .all(|boundary| boundary.test_only || boundary.unresolved_functions.is_empty())
        });
    let passed = complete
        && files.iter().all(|file| file.inline_test_bodies.is_empty())
        && files.iter().all(|file| {
            file.sites
                .iter()
                .all(|site| site.test_only || site.style_present)
        });
    Ok(DocumentationReport {
        schema_version: 1,
        root: layout.root,
        entries: layout.entries,
        macro_expansion_resolved: false,
        complete,
        passed,
        files,
        diagnostics,
    })
}

/// Parsed visitor tied to the original bytes and every incoming module context.
struct DocumentationWalker<'a> {
    /// Immutable source-coordinate index.
    source: &'a Source,
    /// Existing module/cfg/test evidence for this physical source.
    file: &'a FileReport,
    /// Prefix stripped by Syn before assigning byte positions.
    offset: usize,
    /// Documentation requirements found in source order.
    sites: Vec<DocumentationSite>,
    /// All encountered macro boundaries and unresolved authored function tokens.
    macros: Vec<MacroBoundary>,
    /// Avoid duplicate findings when nested macro fragments overlap.
    seen: BTreeSet<(usize, usize, String)>,
    /// Distinguishes ordinary parsed items from source macro fragments.
    origin: &'static str,
    /// Bound macro fragment and ordinary visitor recursion.
    depth: usize,
    /// Fail-closed limitations shared with the module graph.
    diagnostics: &'a mut Vec<crate::Diagnostic>,
    /// Parsed macro fragments can contribute cfg/test owners absent from the graph.
    local_test_spans: Vec<SourceSpan>,
    /// Placement findings are independent of test-function documentation exemptions.
    inline_test_bodies: Vec<SourceSpan>,
}

/// One source function keyword and its preceding same-level authored prefix.
struct MacroFunction {
    /// Original keyword coordinates, never coordinates of a synthetic signature.
    span: SourceSpan,
    /// Authored function name, including a literal `$` for a metavariable name.
    name: String,
    /// Attribute/visibility prefix that Syn can validate independently of metavariables.
    prefix: TokenStream,
}

impl DocumentationWalker<'_> {
    /// Converts parser coordinates without normalizing BOM, shebang, or newlines.
    fn span(&self, span: proc_macro2::Span) -> Option<SourceSpan> {
        let bytes = span.byte_range();
        self.source.span(
            bytes.start.checked_add(self.offset)?,
            bytes.end.checked_add(self.offset)?,
        )
    }

    /// Requires every incoming context to exclude this syntax from production.
    fn test_only(&self, span: &SourceSpan) -> bool {
        let structural = self
            .file
            .test_audit
            .sites
            .iter()
            .any(|site| site.disposition == "test_body" && contains(&site.span, span));
        structural
            || self
                .local_test_spans
                .iter()
                .any(|owner| contains(owner, span))
            || (!self.file.contexts.is_empty()
                && self.file.contexts.iter().all(|context| {
                    context
                        .regions
                        .iter()
                        .filter(|region| contains(&region.span, span))
                        .max_by_key(|region| region.depth)
                        .is_some_and(|region| region.test_only)
                }))
    }

    /// Adds one requirement, retaining alternative rustdoc forms as distinct evidence.
    fn record(
        &mut self,
        span: SourceSpan,
        kind: &str,
        name: &str,
        attrs: &[Attribute],
        module: bool,
    ) {
        if !self
            .seen
            .insert((span.start.byte, span.end.byte, kind.into()))
        {
            return;
        }
        let mut style_present = false;
        let mut rustdoc_present = false;
        let mut computed_documentation = false;
        for attr in attrs {
            let correct_placement = matches!(attr.style, AttrStyle::Inner(_)) == module;
            if attr.path().is_ident("cfg_attr") && conditional_doc(&attr.meta) {
                computed_documentation = true;
            }
            if !attr.path().is_ident("doc") {
                continue;
            }
            let Meta::NameValue(value) = &attr.meta else {
                continue;
            };
            let syn::Expr::Lit(value) = &value.value else {
                computed_documentation = true;
                continue;
            };
            let syn::Lit::Str(value) = &value.lit else {
                computed_documentation = true;
                continue;
            };
            if value.value().trim().is_empty() {
                continue;
            }
            rustdoc_present = true;
            if let Some(span) = self.span(attr.span()) {
                let spelling = self.source.text[span.start.byte..span.end.byte].trim_start();
                style_present |=
                    correct_placement && spelling.starts_with(if module { "//!" } else { "///" });
            }
        }
        self.sites.push(DocumentationSite {
            test_only: self.test_only(&span),
            span,
            kind: kind.into(),
            name: name.into(),
            origin: self.origin.into(),
            required_style: if module { "//!" } else { "///" },
            style_present,
            rustdoc_present,
            computed_documentation,
        });
    }

    /// Records a module definition; a bodyless external declaration is not its body.
    fn record_module(&mut self, span: SourceSpan, name: &str, attrs: &[Attribute]) {
        self.record(span, "module", name, attrs, true);
    }

    /// Records an authored function or method independently of its visibility.
    fn function(
        &mut self,
        span: proc_macro2::Span,
        kind: &str,
        name: &syn::Ident,
        attrs: &[Attribute],
    ) {
        self.remember_test_owner(span, attrs);
        if let Some(span) = self.span(span) {
            self.record(span, kind, &name.to_string(), attrs, false);
        } else {
            self.issue(
                "unresolved_documentation_span",
                "function coordinates do not map to source",
            );
        }
    }

    /// Records explicit harness/cfg ownership for parsed source macro fragments too.
    fn remember_test_owner(&mut self, span: proc_macro2::Span, attrs: &[Attribute]) {
        if (attrs
            .iter()
            .any(|attr| attr.path().is_ident("test") || attr.path().is_ident("bench"))
            || crate::cfg::attributes(attrs).0.test_only())
            && let Some(span) = self.span(span)
        {
            self.local_test_spans.push(span);
        }
    }

    /// Retains a source limitation instead of inferring an empty successful audit.
    fn issue(&mut self, code: &str, reason: &str) {
        self.diagnostics.push(crate::Diagnostic {
            code: code.into(),
            file: self.file.path.clone(),
            span: None,
            reason: reason.into(),
        });
    }

    /// Inspects valid Rust item fragments within macro groups, with a depth bound.
    fn macro_fragments(&mut self, tokens: TokenStream, functions: &mut Vec<MacroFunction>) {
        if self.depth >= 128 {
            self.issue("documentation_traversal_limit", "macro nesting exceeds 128");
            return;
        }
        self.depth += 1;
        let trees: Vec<_> = tokens.clone().into_iter().collect();
        for (index, pair) in trees.windows(2).enumerate() {
            if matches!(&pair[0], TokenTree::Ident(name) if name == "fn")
                && (matches!(&pair[1], TokenTree::Ident(_))
                    || matches!(&pair[1], TokenTree::Punct(punct) if punct.as_char() == '$'))
                && let Some(span) = self.span(pair[0].span())
            {
                let start = trees[..index].iter().rposition(|token| {
                    matches!(token, TokenTree::Punct(p) if p.as_char() == ';')
                        || matches!(token, TokenTree::Group(group) if group.delimiter() == proc_macro2::Delimiter::Brace)
                }).map_or(0, |previous| previous + 1);
                let name = match &pair[1] {
                    TokenTree::Ident(name) => name.to_string(),
                    _ => trees
                        .get(index + 2)
                        .map_or_else(|| "$?".into(), |name| format!("${name}")),
                };
                functions.push(MacroFunction {
                    span,
                    name,
                    prefix: trees[start..index].iter().cloned().collect(),
                });
            }
        }
        if let Ok(parsed) = syn::parse2::<syn::File>(tokens) {
            let previous = self.origin;
            self.origin = "parseable_macro_fragment";
            for item in &parsed.items {
                self.visit_item(item);
            }
            self.origin = previous;
        }
        for tree in trees {
            if let TokenTree::Group(group) = tree {
                self.macro_fragments(group.stream(), functions);
            }
        }
        self.depth -= 1;
    }

    /// Separates parseable authored fragments from unresolved template functions.
    fn macro_boundary(&mut self, mac: &syn::Macro, span: proc_macro2::Span, name: String) {
        let Some(span) = self.span(span) else {
            self.issue(
                "unresolved_documentation_span",
                "macro coordinates do not map to source",
            );
            return;
        };
        let mut functions = Vec::new();
        self.macro_fragments(mac.tokens.clone(), &mut functions);
        functions.retain(|token| {
            !self
                .sites
                .iter()
                .any(|site| site.kind != "module" && contains(&site.span, &token.span))
        });
        let mut unresolved_functions = Vec::new();
        for function in functions {
            // Only the authored prefix is parsed here. The fixed dummy signature
            // validates attributes/visibility without expanding or guessing `$name`.
            let (mut prefix, computed_documentation) = without_template_doc_values(function.prefix);
            prefix.extend(
                "fn __documentation_prefix_probe() {}"
                    .parse::<TokenStream>()
                    .expect("fixed valid Rust tokens"),
            );
            if let Ok(parsed) = syn::parse2::<syn::ItemFn>(prefix) {
                if parsed.attrs.iter().any(|attr| attr.path().is_ident("test"))
                    || crate::cfg::attributes(&parsed.attrs).0.test_only()
                {
                    self.local_test_spans.push(function.span.clone());
                }
                let previous = self.origin;
                self.origin = "authored_macro_template_prefix";
                self.record(
                    function.span,
                    "macro_function_template",
                    &function.name,
                    &parsed.attrs,
                    false,
                );
                if computed_documentation && let Some(site) = self.sites.last_mut() {
                    site.computed_documentation = true;
                }
                self.origin = previous;
            } else {
                unresolved_functions.push(function.span);
            }
        }
        unresolved_functions.sort_by_key(|span| span.start.byte);
        unresolved_functions.dedup_by_key(|span| span.start.byte);
        self.macros.push(MacroBoundary {
            test_only: self.test_only(&span),
            span,
            name,
            unresolved_functions,
        });
    }
}

impl<'ast> Visit<'ast> for DocumentationWalker<'_> {
    /// Bounds ordinary syntax and carries actual cfg ownership into macro fragments.
    fn visit_item(&mut self, item: &'ast syn::Item) {
        if self.depth >= 128 {
            self.issue("documentation_traversal_limit", "item nesting exceeds 128");
            return;
        }
        self.depth += 1;
        let attrs = crate::scanner::item_attributes(item).0;
        self.remember_test_owner(item.span(), attrs);
        visit::visit_item(self, item);
        self.depth -= 1;
    }

    /// Bounds nested expressions independently of module nesting.
    fn visit_expr(&mut self, expression: &'ast syn::Expr) {
        if self.depth >= 128 {
            self.issue(
                "documentation_traversal_limit",
                "expression nesting exceeds 128",
            );
            return;
        }
        self.depth += 1;
        visit::visit_expr(self, expression);
        self.depth -= 1;
    }
    /// Visits ordinary functions and any local function definitions in their bodies.
    fn visit_item_fn(&mut self, item: &'ast syn::ItemFn) {
        self.function(item.span(), "free_function", &item.sig.ident, &item.attrs);
        visit::visit_item_fn(self, item);
    }
    /// Includes inherent and trait-implementation methods, including private helpers.
    fn visit_impl_item_fn(&mut self, item: &'ast syn::ImplItemFn) {
        self.function(item.span(), "impl_method", &item.sig.ident, &item.attrs);
        visit::visit_impl_item_fn(self, item);
    }
    /// Includes required trait signatures and default method implementations.
    fn visit_trait_item_fn(&mut self, item: &'ast syn::TraitItemFn) {
        self.function(item.span(), "trait_method", &item.sig.ident, &item.attrs);
        visit::visit_trait_item_fn(self, item);
    }
    /// Includes authored foreign function declarations even without a Rust body.
    fn visit_foreign_item_fn(&mut self, item: &'ast syn::ForeignItemFn) {
        self.function(
            item.span(),
            "foreign_function",
            &item.sig.ident,
            &item.attrs,
        );
        visit::visit_foreign_item_fn(self, item);
    }
    /// Requires inner documentation on inline production module bodies.
    fn visit_item_mod(&mut self, item: &'ast syn::ItemMod) {
        let start = self.sites.len();
        if item.content.is_some()
            && let Some(span) = self.span(item.span())
        {
            self.record_module(span, &item.ident.to_string(), &item.attrs);
        }
        visit::visit_item_mod(self, item);
        if item.content.is_some()
            && let Some(span) = self.span(item.span())
            && self.test_only(&span)
            && !self.test_only(&self.source.whole_span())
            && self.sites[start..].iter().any(|site| site.kind != "module")
        {
            self.inline_test_bodies.push(span);
        }
    }
    /// Records named macro_rules definitions and ordinary item invocations.
    fn visit_item_macro(&mut self, item: &'ast syn::ItemMacro) {
        self.macro_boundary(
            &item.mac,
            item.span(),
            item.ident
                .as_ref()
                .map(ToString::to_string)
                .unwrap_or_else(|| macro_name(&item.mac)),
        );
    }
    /// Covers associated implementation macros without inventing their expansion.
    fn visit_impl_item_macro(&mut self, item: &'ast syn::ImplItemMacro) {
        self.macro_boundary(&item.mac, item.span(), macro_name(&item.mac));
    }
    /// Covers associated trait macros without mistaking derived methods for source.
    fn visit_trait_item_macro(&mut self, item: &'ast syn::TraitItemMacro) {
        self.macro_boundary(&item.mac, item.span(), macro_name(&item.mac));
    }
    /// Covers macro token bodies in expressions and statement positions.
    fn visit_macro(&mut self, mac: &'ast syn::Macro) {
        self.macro_boundary(mac, mac.span(), macro_name(mac));
    }
}

/// Tests containment using the immutable original-byte coordinate system.
fn contains(outer: &SourceSpan, inner: &SourceSpan) -> bool {
    outer.start.byte <= inner.start.byte && outer.end.byte >= inner.end.byte
}

/// Renders a macro path without adding a token-printing dependency.
fn macro_name(mac: &syn::Macro) -> String {
    mac.path
        .segments
        .iter()
        .map(|segment| segment.ident.to_string())
        .collect::<Vec<_>>()
        .join("::")
}

/// Finds conditional doc attributes in parsed meta rather than string lookalikes.
fn conditional_doc(meta: &Meta) -> bool {
    meta.path().is_ident("doc")
        || (meta.path().is_ident("cfg_attr")
            && crate::cfg::arguments(meta)
                .is_ok_and(|args| args.iter().skip(1).any(conditional_doc)))
}

/// Keeps literal template attributes while recording unexpanded `doc = $value` metadata.
fn without_template_doc_values(tokens: TokenStream) -> (TokenStream, bool) {
    let trees: Vec<_> = tokens.into_iter().collect();
    let mut output = TokenStream::new();
    let mut computed = false;
    let mut index = 0;
    while index < trees.len() {
        if matches!(&trees[index], TokenTree::Punct(punct) if punct.as_char() == '#')
            && let Some(TokenTree::Group(group)) = trees.get(index + 1)
            && group.delimiter() == proc_macro2::Delimiter::Bracket
        {
            let metadata: Vec<_> = group.stream().into_iter().collect();
            if matches!(metadata.first(), Some(TokenTree::Ident(name)) if name == "doc")
                && matches!(metadata.get(1), Some(TokenTree::Punct(punct)) if punct.as_char() == '=')
                && syn::parse2::<Meta>(group.stream()).is_err()
            {
                computed = true;
                index += 2;
                continue;
            }
        }
        output.extend([trees[index].clone()]);
        index += 1;
    }
    (output, computed)
}
