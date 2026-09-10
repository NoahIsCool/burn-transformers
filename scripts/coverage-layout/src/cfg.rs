//! Three-valued configuration reasoning without guessing feature assignments.
//!
//! `False AND Unknown = False` and `True OR Unknown = True`; all other unknown
//! cases remain unknown. Repeated unknown atoms are intentionally not solved
//! symbolically. This sacrifices exclusions, never production source.

use syn::{Attribute, Meta, Token, punctuated::Punctuated};

/// A truth value over all unspecified compiler configurations.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Truth {
    /// False under every assignment considered.
    False,
    /// True under every assignment considered.
    True,
    /// Both truth values remain possible.
    Unknown,
}

impl Truth {
    /// Computes conjunction in conservative three-valued logic.
    fn and(self, other: Self) -> Self {
        match (self, other) {
            (Self::False, _) | (_, Self::False) => Self::False,
            (Self::True, Self::True) => Self::True,
            _ => Self::Unknown,
        }
    }

    /// Computes disjunction in conservative three-valued logic.
    fn or(self, other: Self) -> Self {
        match (self, other) {
            (Self::True, _) | (_, Self::True) => Self::True,
            (Self::False, Self::False) => Self::False,
            _ => Self::Unknown,
        }
    }

    /// Negates a known value while preserving uncertainty.
    fn not(self) -> Self {
        match self {
            Self::False => Self::True,
            Self::True => Self::False,
            Self::Unknown => Self::Unknown,
        }
    }
}

/// The reachability of an item in production and test configurations.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct Reach {
    /// Reachability when `cfg(test)` is false.
    pub(crate) production: Truth,
    /// Reachability when `cfg(test)` is true.
    pub(crate) test: Truth,
    /// An unsupported construct prevents an exclusion proof.
    pub(crate) unresolved: bool,
}

impl Reach {
    /// Starts an unconstrained module-tree entry.
    pub(crate) fn root() -> Self {
        Self {
            production: Truth::True,
            test: Truth::True,
            unresolved: false,
        }
    }

    /// Conjoins this item with its ancestor's reachability.
    pub(crate) fn and(self, other: Self) -> Self {
        Self {
            production: self.production.and(other.production),
            test: self.test.and(other.test),
            unresolved: self.unresolved || other.unresolved,
        }
    }

    /// Indicates a proof of absence in production and possible test presence.
    pub(crate) fn test_only(self) -> bool {
        !self.unresolved && self.production == Truth::False && self.test != Truth::False
    }

    /// Explains why the item is retained or a test-only candidate.
    pub(crate) fn reason(self) -> &'static str {
        if self.unresolved {
            "unresolved syntax or module context; retained"
        } else if self.test_only() {
            "cfg excludes the item with test=false and permits it with test=true"
        } else if self.production == Truth::False && self.test == Truth::False {
            "disabled in both modes, not proven specifically test-only; retained"
        } else if self.production == Truth::Unknown {
            "production reachability depends on unspecified cfg atoms; retained"
        } else {
            "production-reachable source; retained"
        }
    }
}

/// Parses comma-separated meta arguments without interpreting token strings.
pub(crate) fn arguments(meta: &Meta) -> syn::Result<Punctuated<Meta, Token![,]>> {
    match meta {
        Meta::List(list) => list.parse_args_with(Punctuated::parse_terminated),
        _ => Err(syn::Error::new_spanned(
            meta,
            "expected a parenthesized attribute",
        )),
    }
}

/// Evaluates a supported cfg predicate for one value of the built-in test atom.
fn predicate(meta: &Meta, test: bool) -> syn::Result<Truth> {
    match meta {
        Meta::Path(path) if path.is_ident("test") => {
            Ok(if test { Truth::True } else { Truth::False })
        }
        Meta::Path(_) | Meta::NameValue(_) => Ok(Truth::Unknown),
        Meta::List(list) => {
            let args = arguments(meta)?;
            if list.path.is_ident("all") {
                args.iter().try_fold(Truth::True, |value, arg| {
                    Ok(value.and(predicate(arg, test)?))
                })
            } else if list.path.is_ident("any") {
                args.iter().try_fold(Truth::False, |value, arg| {
                    Ok(value.or(predicate(arg, test)?))
                })
            } else if list.path.is_ident("not") && args.len() == 1 {
                Ok(predicate(&args[0], test)?.not())
            } else {
                Err(syn::Error::new_spanned(
                    meta,
                    "unsupported or malformed cfg predicate",
                ))
            }
        }
    }
}

/// Calculates the reachability constraint contributed by a meta attribute.
fn attribute(meta: &Meta, test: bool) -> syn::Result<Truth> {
    if meta.path().is_ident("cfg") {
        let args = arguments(meta)?;
        if args.len() != 1 {
            return Err(syn::Error::new_spanned(meta, "cfg requires one predicate"));
        }
        predicate(&args[0], test)
    } else if meta.path().is_ident("cfg_attr") {
        let args = arguments(meta)?;
        if args.len() < 2 {
            return Err(syn::Error::new_spanned(
                meta,
                "cfg_attr requires a predicate and attribute",
            ));
        }
        let condition = predicate(&args[0], test)?;
        let applied = args.iter().skip(1).try_fold(Truth::True, |value, nested| {
            Ok::<_, syn::Error>(value.and(attribute(nested, test)?))
        })?;
        Ok(condition.not().or(applied))
    } else {
        Ok(Truth::True)
    }
}

/// Combines cfg attributes and retains malformed predicates with diagnostics.
pub(crate) fn attributes(attrs: &[Attribute]) -> (Reach, Vec<String>) {
    let mut reach = Reach::root();
    let mut issues = Vec::new();
    for attr in attrs {
        match (attribute(&attr.meta, false), attribute(&attr.meta, true)) {
            (Ok(production), Ok(test)) => {
                reach = reach.and(Reach {
                    production,
                    test,
                    unresolved: false,
                });
            }
            (left, right) => {
                reach.unresolved = true;
                if let Some(error) = left.err().or_else(|| right.err()) {
                    issues.push(error.to_string());
                }
            }
        }
    }
    (reach, issues)
}

/// Recognizes structural test dependence without changing exclusion candidates.
///
/// Bare harness attributes identify test bodies. Pure cfg is evaluated in both
/// modes, retaining uncertainty over other atoms. Conditional attributes that
/// mention test/bench remain unresolved: an attribute can change expansion or
/// module identity even when it does not change ordinary cfg reachability.
pub(crate) fn test_audit_attribute(attr: &Attribute) -> Option<&'static str> {
    let meta = &attr.meta;
    if meta.path().is_ident("test") || meta.path().is_ident("bench") {
        return Some("test_body");
    }
    if !meta.path().is_ident("cfg") && !meta.path().is_ident("cfg_attr") {
        return None;
    }
    let Ok(args) = arguments(meta) else {
        return Some("unresolved_test_dependent");
    };
    if args.is_empty() || (meta.path().is_ident("cfg_attr") && args.len() < 2) {
        return Some("unresolved_test_dependent");
    }
    if !args.iter().any(mentions_test) {
        return None;
    }
    if meta.path().is_ident("cfg_attr") {
        return Some("unresolved_test_dependent");
    }
    match (attribute(meta, false), attribute(meta, true)) {
        (Ok(Truth::True), _) | (_, Ok(Truth::False)) => None,
        (Ok(Truth::False), Ok(_)) => Some("test_body"),
        _ => Some("unresolved_test_dependent"),
    }
}

/// Searches parsed attribute meta, never strings/comments or opaque macro text.
fn mentions_test(meta: &Meta) -> bool {
    if meta.path().is_ident("test") || meta.path().is_ident("bench") {
        return true;
    }
    matches!(meta, Meta::List(_))
        && arguments(meta).map_or(true, |args| args.iter().any(mentions_test))
}

/// Allows only known non-expanding attributes on a bodyless module declaration.
pub(crate) fn plain_external_attributes(attrs: &[Attribute]) -> bool {
    attrs.iter().all(|attr| {
        [
            "cfg", "path", "doc", "allow", "warn", "deny", "forbid", "expect",
        ]
        .iter()
        .any(|name| attr.path().is_ident(name))
    })
}

/// Collects literal module paths and identifies conditional path ambiguity.
pub(crate) fn paths(attrs: &[Attribute]) -> (Vec<String>, bool) {
    let mut result = Vec::new();
    let mut unresolved = false;
    for attr in attrs {
        collect_paths(&attr.meta, false, &mut result, &mut unresolved);
    }
    (result, unresolved)
}

/// Finds path attributes recursively; conditional paths are never guessed.
fn collect_paths(meta: &Meta, conditional: bool, paths: &mut Vec<String>, unresolved: &mut bool) {
    if meta.path().is_ident("path") {
        *unresolved |= conditional;
        if let Meta::NameValue(value) = meta
            && let syn::Expr::Lit(value) = &value.value
            && let syn::Lit::Str(value) = &value.lit
        {
            paths.push(value.value());
        } else {
            *unresolved = true;
        }
    } else if meta.path().is_ident("cfg_attr") {
        match arguments(meta) {
            Ok(args) => {
                for nested in args.iter().skip(1) {
                    collect_paths(nested, true, paths, unresolved);
                }
            }
            Err(_) => *unresolved = true,
        }
    }
}
