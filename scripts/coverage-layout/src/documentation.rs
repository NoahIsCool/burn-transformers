//! Source-only enforcement of authored Rust documentation style.
//!
//! Syn supplies item and attribute boundaries; the existing classifier supplies
//! all module-reference contexts and structural test ownership. Literal `///`
//! and `//!` are distinguished from equivalent rustdoc attributes. Macro item
//! fragments are inspected when they parse; function templates, Verus proof
//! and specification functions, and member-free declaration templates are
//! checked through their authored attribute and visibility prefix. Unresolved
//! authored function tokens and declaration templates with members remain
//! explicit review blockers, never silently generated or documented code.

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
    /// True only when complete and every required site uses its required style.
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
    /// Authored documentation sites, with the scope selected by the report schema.
    pub sites: Vec<DocumentationSite>,
    /// Macro boundaries, including successfully parsed fragments and opaque tokens.
    pub macros: Vec<MacroBoundary>,
    /// Inline test-module bodies that must move to a sibling test source.
    pub inline_test_bodies: Vec<SourceSpan>,
    /// Schema 2 only: all observed entry contexts are conventional integration harnesses.
    ///
    /// This exempts nested harness tests from unit-test relocation only. It is
    /// entry-path policy, not cfg evidence, and never exempts documentation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub test_harness_only: Option<bool>,
}

/// One real syntax node; visibility never exempts a production helper.
#[derive(Debug, Serialize)]
pub struct DocumentationSite {
    /// Source span including attributes and function/module body.
    pub span: SourceSpan,
    /// Parsed declaration kind; schema 2 additionally includes types and members.
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
    /// Whether the macro is test-owned; schema 2 still audits its authored items.
    pub test_only: bool,
    /// Unresolved authored `fn name` / `fn $name` tokens, excluding function-pointer types.
    pub unresolved_functions: Vec<SourceSpan>,
    /// Schema 2 only: opaque named non-function declarations requiring review.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub unresolved_items: Option<Vec<SourceSpan>>,
}

/// Selects legacy production checks or comprehensive authored-item requirements.
#[derive(Clone, Copy, Eq, PartialEq)]
enum AuditMode {
    /// Preserve schema 1, production function/module scope, and test exemptions.
    Production,
    /// Require line docs for authored types, members, constants, and tests too.
    Standards,
}

/// Audits the parsed module graph while preserving its source and cfg evidence.
pub fn audit_documentation(
    root: &Path,
    entries: &[PathBuf],
) -> Result<DocumentationReport, String> {
    audit(root, entries, AuditMode::Production)
}

/// Audits all parsed authored declarations, including private members and tests.
///
/// Schema 2 preserves source coordinates and macro limitations from the legacy
/// audit. Imports, implementation blocks, generic parameters, and local bindings
/// introduce no separate documentation requirement. Declaration templates with
/// members or metavariable names remain explicit blockers; compiler-generated
/// items are outside this audit.
pub fn audit_standards(root: &Path, entries: &[PathBuf]) -> Result<DocumentationReport, String> {
    audit(root, entries, AuditMode::Standards)
}

/// Shares exact source snapshots and traversal between the versioned policies.
fn audit(root: &Path, entries: &[PathBuf], mode: AuditMode) -> Result<DocumentationReport, String> {
    let layout = crate::classify(root, entries)?;
    let harness_contexts: BTreeSet<_> = layout
        .entries
        .iter()
        .enumerate()
        .filter(|(_, entry)| integration_entry(entry))
        .map(|(index, _)| format!("entry[{index}]"))
        .collect();
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
        let test_harness_only = mode == AuditMode::Standards
            && !file.contexts.is_empty()
            && file.contexts.iter().all(|context| {
                harness_contexts.contains(context.module.split("::").next().unwrap_or_default())
            });
        let mut walker = DocumentationWalker {
            mode,
            test_harness_only,
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
            test_harness_only: (mode == AuditMode::Standards).then_some(test_harness_only),
        });
    }
    let complete = diagnostics
        .iter()
        .all(|issue| issue.code == "duplicate_context")
        && files.iter().all(|file| {
            file.macros.iter().all(|boundary| {
                (mode == AuditMode::Production && boundary.test_only)
                    || (boundary.unresolved_functions.is_empty()
                        && boundary.unresolved_items.as_ref().is_none_or(Vec::is_empty))
            })
        });
    let passed = complete
        && files.iter().all(|file| file.inline_test_bodies.is_empty())
        && files.iter().all(|file| {
            file.sites
                .iter()
                .all(|site| (mode == AuditMode::Production && site.test_only) || site.style_present)
        });
    Ok(DocumentationReport {
        schema_version: if mode == AuditMode::Production { 1 } else { 2 },
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
    /// Controls declaration scope while keeping the legacy JSON contract stable.
    mode: AuditMode,
    /// Relocation exemption for sources referenced only from explicit integration harnesses.
    test_harness_only: bool,
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

    /// Adds a non-function declaration only in the comprehensive standards mode.
    fn declaration(
        &mut self,
        span: proc_macro2::Span,
        kind: &str,
        name: &str,
        attrs: &[Attribute],
    ) {
        if self.mode != AuditMode::Standards {
            return;
        }
        self.remember_test_owner(span, attrs);
        if let Some(span) = self.span(span) {
            self.record(span, kind, name, attrs, false);
        } else {
            self.issue(
                "unresolved_documentation_span",
                "declaration coordinates do not map to source",
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
    fn macro_fragments(
        &mut self,
        tokens: TokenStream,
        functions: &mut Vec<MacroFunction>,
        opaque_items: &mut Vec<SourceSpan>,
    ) {
        if self.depth >= 128 {
            self.issue("documentation_traversal_limit", "macro nesting exceeds 128");
            return;
        }
        self.depth += 1;
        if self.mode == AuditMode::Standards
            && let Ok(parsed) = syn::parse2::<syn::File>(tokens.clone())
        {
            let previous = self.origin;
            self.origin = "parseable_macro_fragment";
            for item in &parsed.items {
                self.visit_item(item);
            }
            self.origin = previous;
            self.depth -= 1;
            return;
        }
        if self.mode == AuditMode::Standards {
            // The braces are only a parser envelope. Every visited declaration
            // keeps the original token coordinates inside the authored group.
            let envelope = TokenTree::Group(proc_macro2::Group::new(
                proc_macro2::Delimiter::Brace,
                tokens.clone(),
            ));
            if let Ok(parsed) = syn::parse2::<syn::Block>(TokenStream::from(envelope)) {
                let previous = self.origin;
                self.origin = "parseable_macro_fragment";
                self.visit_block(&parsed);
                self.origin = previous;
                self.depth -= 1;
                return;
            }
        }
        let trees: Vec<_> = tokens.clone().into_iter().collect();
        for (index, pair) in trees.windows(2).enumerate() {
            if self.mode == AuditMode::Standards
                && let TokenTree::Ident(keyword) = &pair[0]
                && matches!(
                    keyword.to_string().as_str(),
                    "struct" | "enum" | "union" | "trait" | "type" | "static" | "const"
                )
                && (matches!(&pair[1], TokenTree::Ident(name) if name != "fn")
                    || matches!(&pair[1], TokenTree::Punct(punct) if punct.as_char() == '$'))
                && !keyword_continues_type(&trees, index)
                && let Some(span) = self.span(pair[0].span())
                && !self.declaration_template(&trees, index, span.clone())
            {
                opaque_items.push(span);
            }
            if matches!(&pair[0], TokenTree::Ident(name) if name == "fn")
                && (matches!(&pair[1], TokenTree::Ident(_))
                    || matches!(&pair[1], TokenTree::Punct(punct) if punct.as_char() == '$'))
                && let Some(span) = self.span(pair[0].span())
            {
                let start = prefix_start(&trees, index);
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
        if self.mode == AuditMode::Production
            && let Ok(parsed) = syn::parse2::<syn::File>(tokens)
        {
            let previous = self.origin;
            self.origin = "parseable_macro_fragment";
            for item in &parsed.items {
                self.visit_item(item);
            }
            self.origin = previous;
        }
        for tree in trees {
            if let TokenTree::Group(group) = tree {
                self.macro_fragments(group.stream(), functions, opaque_items);
            }
        }
        self.depth -= 1;
    }

    /// Resolves a fixed-name declaration without members from its authored prefix.
    ///
    /// A constant, static, or type alias ends at its own `;` and declares no
    /// members, so its literal documentation is decided entirely by the
    /// attributes and visibility before the keyword. Those parse independently
    /// of metavariables in its type or initializer, exactly as a function
    /// template's prefix does. Returns `false`, leaving the declaration an
    /// opaque review blocker, for types with bodies, metavariable names, and
    /// prefixes that are not ordinary attributes and visibility.
    fn declaration_template(
        &mut self,
        trees: &[TokenTree],
        index: usize,
        span: SourceSpan,
    ) -> bool {
        let kind = match &trees[index] {
            TokenTree::Ident(keyword) if keyword == "const" => "constant_template",
            TokenTree::Ident(keyword) if keyword == "static" => "static_template",
            TokenTree::Ident(keyword) if keyword == "type" => "type_alias_template",
            _ => return false,
        };
        let Some(TokenTree::Ident(name)) = trees.get(index + 1) else {
            return false;
        };
        let prefix: TokenStream = trees[prefix_start(trees, index)..index]
            .iter()
            .cloned()
            .collect();
        let (mut prefix, computed_documentation) = without_template_doc_values(prefix);
        let Ok(probe) = "struct __documentation_prefix_probe;".parse::<TokenStream>() else {
            self.issue(
                "invalid_prefix_probe",
                "fixed declaration probe did not tokenize",
            );
            return false;
        };
        prefix.extend(probe);
        let attrs = syn::parse2::<syn::File>(prefix).ok().and_then(|file| {
            let mut items = file.items.into_iter();
            match (items.next(), items.next()) {
                (Some(syn::Item::Struct(item)), None) => Some(item.attrs),
                _ => None,
            }
        });
        let Some(attrs) = attrs else {
            return false;
        };
        if crate::cfg::attributes(&attrs).0.test_only() {
            self.local_test_spans.push(span.clone());
        }
        let previous = self.origin;
        self.origin = "authored_macro_template_prefix";
        let recorded = self.sites.len();
        self.record(span, kind, &name.to_string(), &attrs, false);
        if computed_documentation && self.sites.len() > recorded {
            self.sites[recorded].computed_documentation = true;
        }
        self.origin = previous;
        true
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
        let mut opaque_items = Vec::new();
        self.macro_fragments(mac.tokens.clone(), &mut functions, &mut opaque_items);
        functions.retain(|token| {
            !self
                .sites
                .iter()
                .any(|site| function_kind(&site.kind) && contains(&site.span, &token.span))
        });
        // Verus writes proof and specification functions inside `verus!` with
        // mode words (`proof fn`, `pub open spec fn`) that Syn cannot parse.
        // Only those words are dropped; attributes, docs, and visibility are
        // still validated exactly as for any other macro function prefix.
        let verus = name.rsplit("::").next() == Some("verus");
        let mut unresolved_functions = Vec::new();
        for function in functions {
            let prefix = if verus {
                without_verus_modes(function.prefix)
            } else {
                function.prefix
            };
            // Only the authored prefix is parsed here. The fixed dummy signature
            // validates attributes/visibility without expanding or guessing `$name`.
            let (mut prefix, computed_documentation) = without_template_doc_values(prefix);
            let signature = "fn __documentation_prefix_probe() {}".parse::<TokenStream>();
            let Ok(signature) = signature else {
                self.issue(
                    "invalid_prefix_probe",
                    "fixed documentation probe did not tokenize",
                );
                return;
            };
            prefix.extend(signature);
            let parsed = if self.mode == AuditMode::Standards {
                // Property-test macros accept macro-level inner configuration
                // attributes before their first function. A file envelope
                // separates those from the function's own outer documentation.
                syn::parse2::<syn::File>(prefix).ok().and_then(|file| {
                    let mut items = file.items.into_iter();
                    match (items.next(), items.next()) {
                        (Some(syn::Item::Fn(function)), None) => Some(function),
                        _ => None,
                    }
                })
            } else {
                syn::parse2::<syn::ItemFn>(prefix).ok()
            };
            if let Some(parsed) = parsed {
                if parsed.attrs.iter().any(|attr| attr.path().is_ident("test"))
                    || crate::cfg::attributes(&parsed.attrs).0.test_only()
                {
                    self.local_test_spans.push(function.span.clone());
                }
                let previous = self.origin;
                self.origin = if verus {
                    "verus_function_prefix"
                } else {
                    "authored_macro_template_prefix"
                };
                self.record(
                    function.span,
                    if verus {
                        "verus_function"
                    } else {
                        "macro_function_template"
                    },
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
        opaque_items.sort_by_key(|span| span.start.byte);
        opaque_items.dedup_by_key(|span| span.start.byte);
        self.macros.push(MacroBoundary {
            test_only: self.test_only(&span),
            span,
            name,
            unresolved_functions,
            unresolved_items: (self.mode == AuditMode::Standards).then_some(opaque_items),
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
        let declaration = match item {
            syn::Item::Struct(item) => Some(("struct", &item.ident)),
            syn::Item::Enum(item) => Some(("enum", &item.ident)),
            syn::Item::Union(item) => Some(("union", &item.ident)),
            syn::Item::Trait(item) => Some(("trait", &item.ident)),
            syn::Item::TraitAlias(item) => Some(("trait_alias", &item.ident)),
            syn::Item::Type(item) => Some(("type_alias", &item.ident)),
            syn::Item::Const(item) => Some(("constant", &item.ident)),
            syn::Item::Static(item) => Some(("static", &item.ident)),
            _ => None,
        };
        if let Some((kind, name)) = declaration {
            self.declaration(item.span(), kind, &name.to_string(), attrs);
        }
        visit::visit_item(self, item);
        self.depth -= 1;
    }
    /// Checks trait-implementation associated constants and type definitions.
    fn visit_impl_item(&mut self, item: &'ast syn::ImplItem) {
        match item {
            syn::ImplItem::Const(item) => self.declaration(
                item.span(),
                "impl_constant",
                &item.ident.to_string(),
                &item.attrs,
            ),
            syn::ImplItem::Type(item) => self.declaration(
                item.span(),
                "impl_type",
                &item.ident.to_string(),
                &item.attrs,
            ),
            _ => {}
        }
        visit::visit_impl_item(self, item);
    }
    /// Checks required/default associated constants and types in trait contracts.
    fn visit_trait_item(&mut self, item: &'ast syn::TraitItem) {
        match item {
            syn::TraitItem::Const(item) => self.declaration(
                item.span(),
                "trait_constant",
                &item.ident.to_string(),
                &item.attrs,
            ),
            syn::TraitItem::Type(item) => self.declaration(
                item.span(),
                "trait_type",
                &item.ident.to_string(),
                &item.attrs,
            ),
            _ => {}
        }
        visit::visit_trait_item(self, item);
    }
    /// Checks externally defined statics and opaque types alongside foreign functions.
    fn visit_foreign_item(&mut self, item: &'ast syn::ForeignItem) {
        match item {
            syn::ForeignItem::Static(item) => self.declaration(
                item.span(),
                "foreign_static",
                &item.ident.to_string(),
                &item.attrs,
            ),
            syn::ForeignItem::Type(item) => self.declaration(
                item.span(),
                "foreign_type",
                &item.ident.to_string(),
                &item.attrs,
            ),
            _ => {}
        }
        visit::visit_foreign_item(self, item);
    }
    /// Requires each enum variant to explain its own case independently of its enum.
    fn visit_variant(&mut self, item: &'ast syn::Variant) {
        self.declaration(item.span(), "variant", &item.ident.to_string(), &item.attrs);
        visit::visit_variant(self, item);
    }
    /// Uses declared names for struct, union, and record-variant member findings.
    fn visit_fields_named(&mut self, fields: &'ast syn::FieldsNamed) {
        for field in &fields.named {
            let name = field
                .ident
                .as_ref()
                .map_or_else(String::new, ToString::to_string);
            self.declaration(field.span(), "field", &name, &field.attrs);
        }
        visit::visit_fields_named(self, fields);
    }
    /// Uses zero-based positions for tuple-struct and tuple-variant member findings.
    fn visit_fields_unnamed(&mut self, fields: &'ast syn::FieldsUnnamed) {
        for (index, field) in fields.unnamed.iter().enumerate() {
            self.declaration(
                field.span(),
                "tuple_field",
                &index.to_string(),
                &field.attrs,
            );
        }
        visit::visit_fields_unnamed(self, fields);
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
            && !self.test_harness_only
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
        if let Some(name) = &item.ident {
            self.declaration(
                item.span(),
                "macro_definition",
                &name.to_string(),
                &item.attrs,
            );
        }
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

/// Identifies function findings without treating enclosing type spans as functions.
fn function_kind(kind: &str) -> bool {
    matches!(
        kind,
        "free_function"
            | "impl_method"
            | "trait_method"
            | "foreign_function"
            | "macro_function_template"
            | "verus_function"
    )
}

/// Returns where the authored prefix of the declaration keyword at `index` begins.
///
/// The prefix is everything at the same token level after the previous `;` or
/// braced body: the declaration's attributes, documentation, and visibility.
fn prefix_start(trees: &[TokenTree], index: usize) -> usize {
    trees[..index]
        .iter()
        .rposition(|token| {
            matches!(token, TokenTree::Punct(p) if p.as_char() == ';')
                || matches!(token, TokenTree::Group(group) if group.delimiter() == proc_macro2::Delimiter::Brace)
        })
        .map_or(0, |previous| previous + 1)
}

/// Whether a declaration keyword is part of a type rather than a declaration.
///
/// Proc-macro tokens split the lifetime `'static` into a `'` punct and the
/// `static` keyword, and a raw pointer spells `*const T`; neither declares an
/// item, so neither may become an opaque declaration finding.
fn keyword_continues_type(trees: &[TokenTree], index: usize) -> bool {
    index
        .checked_sub(1)
        .and_then(|previous| trees.get(previous))
        .is_some_and(|previous| {
            matches!(previous, TokenTree::Punct(punct) if punct.as_char() == '\'' || punct.as_char() == '*')
        })
}

/// Verus mode and modifier words that may precede `fn` inside `verus!`.
const VERUS_FUNCTION_MODIFIERS: [&str; 10] = [
    "axiom",
    "broadcast",
    "closed",
    "exec",
    "ghost",
    "open",
    "proof",
    "spec",
    "tracked",
    "uninterp",
];

/// Removes top-level Verus mode words so Syn can validate the remaining prefix.
///
/// Attribute and visibility groups are kept whole, so a mode word inside an
/// attribute argument is never touched.
fn without_verus_modes(tokens: TokenStream) -> TokenStream {
    tokens
        .into_iter()
        .filter(|token| {
            !matches!(token, TokenTree::Ident(word)
                if VERUS_FUNCTION_MODIFIERS.contains(&word.to_string().as_str()))
        })
        .collect()
}

/// Recognizes explicit repository integration roots, never a helper filename alone.
fn integration_entry(entry: &str) -> bool {
    let path = Path::new(entry);
    matches!(path.components().next(), Some(std::path::Component::Normal(root)) if root == "tests" || root == "scripts")
        && path
            .parent()
            .and_then(Path::file_name)
            .is_some_and(|name| name == "tests")
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
