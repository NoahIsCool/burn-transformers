//! Bounded structural test audit, independent of source exclusion decisions.
//!
//! This visitor sees parsed attributes on files, items, associated items,
//! expressions, and statements. It does not expand macro token streams or
//! attest arbitrary macro I/O. Opaque runtime expansions stay in coverage.
//! A complete audit means all non-opaque parsed syntax was visited within the
//! bounds, not that generated syntax or all possible compiler cfgs were proven.

use serde::Serialize;
use syn::{
    spanned::Spanned,
    visit::{self, Visit},
};

use crate::{
    cfg,
    source::{Source, SourceSpan},
};

/// Structural audit version; old classifier reports cannot qualify promotion.
const VERSION: u32 = 1;
/// Independent bound includes expression/statement depth, not just modules.
const MAX_DEPTH: usize = 128;
/// Bounds emitted audit records even for dense adversarial attribute input.
const MAX_SITES: usize = 65_536;

/// A physical-file structural audit, not a macro expansion or compiler proof.
#[derive(Clone, Debug, Serialize)]
pub struct TestAudit {
    /// Exact version of the audited structural constructs and retention rules.
    pub schema_version: u32,
    /// All parsed non-opaque syntax was visited without a span/depth failure.
    pub complete: bool,
    /// Always false: macro expansion is outside this source-only audit.
    pub macro_expansion_resolved: bool,
    /// Explicit test-dependent owners, including harmless external declarations.
    pub sites: Vec<TestSite>,
}

/// An AST owner whose attributes can introduce test-specific source.
#[derive(Clone, Debug, Serialize)]
pub struct TestSite {
    /// Complete owner span; no partial line is excluded using this information.
    pub span: SourceSpan,
    /// Exact triggering attribute span within the owner.
    pub attribute_span: SourceSpan,
    /// File/item/inline_mod/external_mod/impl_item/trait_item/expr/stmt owner.
    pub owner_kind: String,
    /// test_body, external_declaration, or unresolved_test_dependent.
    pub disposition: String,
}

/// Represents absent/failed parsing without inventing an empty successful audit.
pub(crate) fn unparsed() -> TestAudit {
    TestAudit {
        schema_version: VERSION,
        complete: false,
        macro_expansion_resolved: false,
        sites: Vec::new(),
    }
}

/// Audits exact parsed source bytes with the classifier's BOM/shebang offset.
pub(crate) fn audit(source: &Source, file: Option<&syn::File>, offset: usize) -> TestAudit {
    let Some(file) = file else {
        return unparsed();
    };
    let mut walker = Walker {
        source,
        offset,
        depth: 0,
        owner: source.whole_span(),
        kind: "file",
        plain_external: false,
        result: TestAudit {
            complete: true,
            ..unparsed()
        },
    };
    walker.visit_file(file);
    walker.result
}

/// Traversal state whose owner is restored after every nested syntax node.
struct Walker<'a> {
    /// Immutable source buffer and original-byte coordinate mapping.
    source: &'a Source,
    /// Prefix removed before Syn positions were assigned.
    offset: usize,
    /// Current nested owner depth.
    depth: usize,
    /// Complete current syntax owner, not the last visited attribute.
    owner: SourceSpan,
    /// Stable category interpreted by the bounded Python promotion predicate.
    kind: &'static str,
    /// Only an unambiguous module with non-expanding attributes can be harmless.
    plain_external: bool,
    /// Accumulated positive evidence and unresolved test-dependent sites.
    result: TestAudit,
}

impl Walker<'_> {
    /// Maps a parser span back to original bytes or marks the audit incomplete.
    fn span(&mut self, span: proc_macro2::Span) -> Option<SourceSpan> {
        let range = span.byte_range();
        let mapped = range
            .start
            .checked_add(self.offset)
            .zip(range.end.checked_add(self.offset))
            .and_then(|(start, end)| self.source.span(start, end));
        if mapped.is_none() {
            self.result.complete = false;
        }
        mapped
    }

    /// Visits one owner and restores its parent without trusting unbounded depth.
    fn within(
        &mut self,
        span: proc_macro2::Span,
        kind: &'static str,
        plain_external: bool,
        visit: impl FnOnce(&mut Self),
    ) {
        if self.depth >= MAX_DEPTH {
            self.result.complete = false;
            return;
        }
        let Some(span) = self.span(span) else {
            return;
        };
        let previous = (self.owner.clone(), self.kind, self.plain_external);
        self.owner = span;
        self.kind = kind;
        self.plain_external = plain_external;
        self.depth += 1;
        visit(self);
        self.depth -= 1;
        (self.owner, self.kind, self.plain_external) = previous;
    }
}

impl<'ast> Visit<'ast> for Walker<'_> {
    /// Handles item attributes, distinguishing a declaration from an inline body.
    fn visit_item(&mut self, item: &'ast syn::Item) {
        let (kind, plain) = match item {
            syn::Item::Mod(module) if module.content.is_none() => {
                let (paths, conditional) = cfg::paths(&module.attrs);
                (
                    "external_mod",
                    !conditional
                        && paths.len() <= 1
                        && cfg::plain_external_attributes(&module.attrs),
                )
            }
            syn::Item::Mod(_) => ("inline_mod", false),
            syn::Item::Verbatim(_) => {
                self.result.complete = false;
                return;
            }
            _ => ("item", false),
        };
        self.within(item.span(), kind, plain, |this| {
            visit::visit_item(this, item)
        });
    }

    /// Includes cfg attributes on inherent and trait-implementation methods.
    fn visit_impl_item(&mut self, item: &'ast syn::ImplItem) {
        if matches!(item, syn::ImplItem::Verbatim(_)) {
            self.result.complete = false;
            return;
        }
        self.within(item.span(), "impl_item", false, |this| {
            visit::visit_impl_item(this, item)
        });
    }

    /// Includes cfg attributes on trait methods, defaults, constants and types.
    fn visit_trait_item(&mut self, item: &'ast syn::TraitItem) {
        if matches!(item, syn::TraitItem::Verbatim(_)) {
            self.result.complete = false;
            return;
        }
        self.within(item.span(), "trait_item", false, |this| {
            visit::visit_trait_item(this, item)
        });
    }

    /// Includes attributes on every parsed expression variant via Syn's visitor.
    fn visit_expr(&mut self, expr: &'ast syn::Expr) {
        if matches!(expr, syn::Expr::Verbatim(_)) {
            self.result.complete = false;
            return;
        }
        self.within(expr.span(), "expr", false, |this| {
            visit::visit_expr(this, expr)
        });
    }

    /// Includes local/macro statement attributes before visiting nested bodies.
    fn visit_stmt(&mut self, stmt: &'ast syn::Stmt) {
        self.within(stmt.span(), "stmt", false, |this| {
            visit::visit_stmt(this, stmt)
        });
    }

    /// Records real parsed test-sensitive attributes, not token/string lookalikes.
    fn visit_attribute(&mut self, attr: &'ast syn::Attribute) {
        let Some(mut disposition) = cfg::test_audit_attribute(attr) else {
            return;
        };
        if self.result.sites.len() >= MAX_SITES {
            self.result.complete = false;
            return;
        }
        let Some(attribute_span) = self.span(attr.span()) else {
            return;
        };
        if self.kind == "external_mod" {
            disposition = if self.plain_external && disposition == "test_body" {
                "external_declaration"
            } else {
                "unresolved_test_dependent"
            };
        }
        self.result.sites.push(TestSite {
            span: self.owner.clone(),
            attribute_span,
            owner_kind: self.kind.into(),
            disposition: disposition.into(),
        });
    }
}
