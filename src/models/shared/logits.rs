//! Selection of hidden states before the vocabulary projection.
//!
//! The vocabulary projection `h ↦ h Wᵀ` costs `O(S · d · V)` and dominates a
//! small decoder step, so families project only the requested positions:
//!
//! * [`LogitsSelection::All`] keeps every column `[B, S, d]`;
//! * [`LogitsSelection::Last`] keeps each row's last valid column `[B, 1, d]`.
//!   With a mask this is `t*_b = max {t : m[b, t]}`, computed on device as
//!   `argmax_t (m[b, t] ? t : -1)` and gathered, so no host read is needed;
//! * [`LogitsSelection::Positions`] keeps an explicit ordered column list
//!   `[B, |P|, d]` shared by all rows.
//!
//! Every path copies hidden values without arithmetic, so selection is exact
//! in every dtype and differentiable through Burn's gather/select backward.

use burn::tensor::{Int, Tensor, TensorData, backend::Backend};

use crate::{Error, LogitsSelection, Result, architecture::TokenMask};

use super::plan::{self, LastSelection};

/// Name used in errors for invalid explicit selections.
const SELECTION_NAME: &str = "forward_options.logits.positions";

/// Rejects an explicit selection that is empty or reaches beyond `sequence`.
///
/// Called before any stateful work so an invalid request cannot mutate caches.
///
/// # Errors
///
/// Returns [`Error::Shape`] naming `forward_options.logits.positions`.
pub(crate) fn validate_logits_selection(
    selection: &LogitsSelection,
    sequence: usize,
) -> Result<()> {
    if let LogitsSelection::Positions(positions) = selection
        && !plan::positions_within(positions, sequence)
    {
        return Err(Error::shape(
            SELECTION_NAME,
            format!("a non-empty list of indices below {sequence}"),
            format!("{positions:?}"),
        ));
    }
    Ok(())
}

/// Selects the hidden states whose logits were requested.
///
/// `hidden` is `[B, S, d]` and `mask` uses `true == valid`. A fully masked row
/// selects column `0` under [`LogitsSelection::Last`]; families that permit
/// such rows give them zero hidden states, so the selected state is zero.
///
/// # Errors
///
/// Returns [`Error::Shape`] for an invalid explicit selection and
/// [`Error::Configuration`] when indices exceed Burn's signed index range.
pub(crate) fn select_hidden<B: Backend>(
    hidden: Tensor<B, 3>,
    mask: Option<&TokenMask<B>>,
    selection: &LogitsSelection,
) -> Result<Tensor<B, 3>> {
    let [batch, sequence, width] = hidden.dims();
    validate_logits_selection(selection, sequence)?;
    match (selection, mask) {
        (LogitsSelection::All, _) => Ok(hidden),
        (LogitsSelection::Last, None) => Ok(hidden.narrow(1, sequence - 1, 1)),
        (LogitsSelection::Last, Some(mask)) => {
            let columns = super::input::physical_columns::<B>(batch, sequence, &hidden.device())?;
            // Padding scores -1, so the argmax is the greatest valid column.
            let last = columns
                .mask_fill(mask.as_tensor().clone().bool_not(), -1)
                .argmax(1)
                .unsqueeze_dim::<3>(2)
                .expand([batch, 1, width]);
            Ok(hidden.gather(1, last))
        }
        (LogitsSelection::Positions(positions), _) => select_columns(hidden, positions),
    }
}

/// Selects hidden states when each row's last valid column is known on the host.
///
/// Stateful calls already hold the per-row valid columns, so
/// [`LogitsSelection::Last`] becomes one gather with an uploaded index instead
/// of the arange/mask/argmax chain. `last_columns[b]` must be row `b`'s last
/// valid column (any column for an inactive row, whose states are zero).
///
/// # Errors
///
/// Returns [`Error::Shape`] for an invalid selection or a `last_columns`
/// length different from the batch, and [`Error::Configuration`] for indices
/// beyond Burn's signed index range.
pub(crate) fn select_hidden_with_last_columns<B: Backend>(
    hidden: Tensor<B, 3>,
    last_columns: &[usize],
    selection: &LogitsSelection,
) -> Result<Tensor<B, 3>> {
    let [batch, sequence, width] = hidden.dims();
    validate_logits_selection(selection, sequence)?;
    match selection {
        LogitsSelection::All => Ok(hidden),
        LogitsSelection::Positions(positions) => select_columns(hidden, positions),
        LogitsSelection::Last => match plan::plan_last_selection(last_columns, batch, sequence) {
            LastSelection::Invalid => Err(Error::shape(
                "logits.last_columns",
                format!("{batch} columns below {sequence}"),
                format!("{last_columns:?}"),
            )),
            LastSelection::FinalColumn => Ok(hidden.narrow(1, sequence - 1, 1)),
            LastSelection::Gather => {
                let values = signed(last_columns, "logits.last_columns")?;
                let last = Tensor::<B, 2, Int>::from_data(
                    TensorData::new(values, [batch, 1]),
                    &hidden.device(),
                )
                .unsqueeze_dim::<3>(2)
                .expand([batch, 1, width]);
                Ok(hidden.gather(1, last))
            }
        },
    }
}

/// Selects the explicit column list `positions` from every row.
fn select_columns<B: Backend>(hidden: Tensor<B, 3>, positions: &[usize]) -> Result<Tensor<B, 3>> {
    let values = signed(positions, SELECTION_NAME)?;
    let selector = Tensor::<B, 1, Int>::from_data(
        TensorData::new(values, [positions.len()]),
        &hidden.device(),
    );
    Ok(hidden.select(1, selector))
}

/// Converts host indices to `i64`, reporting values beyond the signed range.
fn signed(indices: &[usize], name: &str) -> Result<Vec<i64>> {
    plan::signed_indices(indices)
        .ok_or_else(|| Error::configuration(name, "index exceeds the signed tensor-index range"))
}

#[cfg(test)]
#[path = "logits_tests.rs"]
mod tests;
