//! Original-token-coordinate hook execution for padded T5 decoder batches.
//!
//! Hook selections retain the caller's rectangular token coordinates. Validity
//! masks broadcast over channel and head axes, preventing edits from restoring
//! padded Q/K/V, head-context, or score-plane entries. They express semantic token
//! validity (`true` means present), independently of causal connectivity.
//!
//! Only eager FP32 softmax exposes score/pattern points; this module performs
//! boolean/integer bookkeeping and never changes attention arithmetic. Logical
//! decoder positions advance on valid tokens alone, so holes and left padding do
//! not alter T5's relative-position distances.

use burn::tensor::Bool;

use super::*;
use crate::intervention::PointValidity;

impl<B: Backend> T5ForConditionalGeneration<B> {
    /// Declares padding invariants without registering extra capture/edit actions.
    pub(super) fn decoder_hook_validity(
        &self,
        input: &Seq2SeqInput<B>,
    ) -> Result<Vec<PointValidity<B>>> {
        let Some(mask) = &input.decoder.token_mask else {
            return Ok(Vec::new());
        };
        let [batch, targets] = input.decoder.token_ids.dims();
        let [source_batch, sources] = input.encoder.token_ids.dims();
        // Preserve the existing contextual input errors inside the hook scope.
        // Invalid shapes must not be transformed or broadcast while making masks.
        let device = self.shared.weight.lazy_device();
        if mask.as_tensor().device() != device
            || input
                .encoder
                .token_mask
                .as_ref()
                .is_some_and(|mask| mask.as_tensor().device() != device)
            || batch == 0
            || targets == 0
            || sources == 0
            || source_batch != batch
            || mask.as_tensor().dims() != [batch, targets]
            || input
                .encoder
                .token_mask
                .as_ref()
                .is_some_and(|mask| mask.as_tensor().dims() != [batch, sources])
        {
            return Ok(Vec::new());
        }
        let target = mask.as_tensor().clone().int().bool();
        let source = input
            .encoder
            .token_mask
            .as_ref()
            .map(|mask| mask.as_tensor().clone().int().bool())
            .unwrap_or_else(|| Tensor::<B, 2, Bool>::ones([batch, sources], &target.device()));
        let hidden_mask = target.clone().unsqueeze_dim::<3>(2);
        let mut validity = vec![
            PointValidity::new(Stack::Decoder.embedding(), hidden_mask.clone())?,
            PointValidity::new(Stack::Decoder.final_norm(), hidden_mask.clone())?,
        ];
        for (index, block) in self.decoder.block.iter().enumerate() {
            let layer = Layer::decoder(index);
            for point in [
                layer.residual_pre(),
                layer.attention_input(),
                layer.attention_output(),
                layer.mlp_input(),
                layer.mlp_pre_activation(),
                layer.mlp_activation(),
                layer.mlp_output(),
                layer.residual_post(),
            ] {
                validity.push(PointValidity::new(point, hidden_mask.clone())?);
            }
            attention_validity(
                &mut validity,
                layer.self_attention(),
                &target,
                &target,
                block.self_attention.self_attention.attention.precision_mode,
            )?;
            if let Some(cross) = &block.cross_attention {
                validity.push(PointValidity::new(
                    layer.cross_attention_input(),
                    hidden_mask.clone(),
                )?);
                validity.push(PointValidity::new(
                    layer.cross_attention_output(),
                    hidden_mask.clone(),
                )?);
                attention_validity(
                    &mut validity,
                    layer.cross_attention(),
                    &target,
                    &source,
                    cross.enc_dec_attention.attention.precision_mode,
                )?;
            }
        }
        Ok(validity)
    }
}

// Axis-aware padding validity for attention projections and score planes.

/// Adds broadcast validity masks for native Q/K/V, head outputs, and visible score planes.
///
/// Query/head masks use `[batch, 1, query, 1]`; key/value masks replace
/// query with key length. Eager score/pattern masks use the outer conjunction
/// `query_valid AND key_valid`, so interventions cannot revive padded entries.
fn attention_validity<B: Backend>(
    validity: &mut Vec<PointValidity<B>>,
    points: AttentionPoints,
    query: &Tensor<B, 2, Bool>,
    key: &Tensor<B, 2, Bool>,
    precision: AttentionPrecisionMode,
) -> Result<()> {
    let [batch, queries] = query.dims();
    let keys = key.dims()[1];
    let query_mask = query.clone().reshape([batch, 1, queries, 1]);
    let key_mask = key.clone().reshape([batch, 1, keys, 1]);
    validity.extend([
        PointValidity::new(points.query(), query_mask.clone())?,
        PointValidity::new(points.key(), key_mask.clone())?,
        PointValidity::new(points.value(), key_mask)?,
        PointValidity::new(points.head_context(), query_mask.clone())?,
    ]);
    if precision == AttentionPrecisionMode::EagerFp32Softmax {
        let plane = query_mask.bool_and(key.clone().reshape([batch, 1, 1, keys]));
        validity.extend([
            PointValidity::new(points.scores(), plane.clone())?,
            PointValidity::new(points.pattern(), plane)?,
        ]);
    }
    Ok(())
}

// Semantic positions ignore padding while preserving rectangular hook coordinates.

/// Valid target tokens advance relative positions; padding never consumes one.
///
/// Position `p[b, t] = Σ_{u < t} m[b, u]` counts the valid tokens strictly
/// before column `t` (an exclusive prefix sum), so a padded column shares the
/// position of the next valid token. It is computed on device as
/// `cumsum(m) - m` in the backend's integer dtype, which is exact and needs
/// no host readback.
pub(super) fn logical_decoder_positions<B: Backend>(
    mask: &TokenMask<B>,
) -> Result<Tensor<B, 2, Int>> {
    let [batch, sequence] = mask.as_tensor().dims();
    if batch == 0 || sequence == 0 {
        return Err(Error::configuration(
            "t5.decoder.token_mask",
            "mask must be nonempty",
        ));
    }
    let valid = mask.as_tensor().clone().int();
    Ok(valid.clone().cumsum(1) - valid)
}
