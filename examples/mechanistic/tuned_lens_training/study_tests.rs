use super::*;

fn tiny_lens() -> Lens<B> {
    let device = Default::default();
    let backbone = GptNeoXModel::init(
        GptNeoXConfig {
            vocab_size: 7,
            hidden_size: 4,
            intermediate_size: 8,
            num_attention_heads: 1,
            num_hidden_layers: 1,
            rotary_pct: 1.0,
            ..Default::default()
        },
        &device,
    )
    .unwrap();
    Lens::new(&backbone)
}

fn batch() -> CachedRecord {
    CachedRecord {
        hidden: vec![Tensor::from_floats(
            [[1.0, 2.0, 3.0, 4.0], [-1.0, 0.5, 2.0, 3.0]],
            &Default::default(),
        )],
        teacher: Tensor::from_floats(
            [
                [1.0, 2.0, 0.0, -1.0, 3.0, 1.0, 0.0],
                [0.0, 1.0, 3.0, 2.0, 0.0, 1.0, -1.0],
            ],
            &Default::default(),
        ),
        tokens: vec![1, 2],
    }
}

#[test]
fn frozen_decoder_propagates_translator_gradients_without_updating_its_weights() {
    let mut lens = tiny_lens();
    let batch = batch();
    let before = lens.frozen_fingerprint();
    let raw = lens
        .unembedding
        .forward(lens.final_norm.forward(batch.hidden[0].clone()));
    assert_eq!(
        max_difference(raw, lens.forward(batch.hidden[0].clone())).unwrap(),
        0.0
    );
    let loss =
        transformers::training::kl_divergence(lens.forward(batch.hidden[0].clone()), batch.teacher)
            .unwrap();
    let (gradients, norm) = gradients(loss.backward(), &lens).unwrap();
    assert!(norm > 0.0);
    assert_eq!(gradients.len(), 2);
    lens = optimizer().step(0.02, lens, gradients);
    assert_eq!(before, lens.frozen_fingerprint());
    assert!(
        lens.translator
            .weight
            .val()
            .abs()
            .max()
            .into_data()
            .to_vec::<f32>()
            .unwrap()[0]
            > 0.0
    );
}

#[test]
fn saved_affine_and_momentum_records_reproduce_the_next_optimizer_step() {
    let directory = tempfile::tempdir().unwrap();
    let settings = Settings {
        base: PathBuf::new(),
        output: directory.path().into(),
        stages: vec![0],
        train_records: 1,
        steps: 2,
        protocol: Value::Null,
    };
    let mut lens = tiny_lens();
    let batch = batch();
    let mut optim = optimizer();
    let loss = transformers::training::kl_divergence(
        lens.forward(batch.hidden[0].clone()),
        batch.teacher.clone(),
    )
    .unwrap();
    let (gradients, _) = gradients(loss.backward(), &lens).unwrap();
    lens = optim.step(0.02, lens, gradients);
    let (restored, restored_optim) = checkpoint(&settings, 0, &lens, &optim).unwrap();
    assert_eq!(
        compare_next_step(lens, optim, restored, restored_optim, &batch, 0, 0.01).unwrap(),
        0.0
    );
}
