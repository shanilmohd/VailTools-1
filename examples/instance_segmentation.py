"""
TODO: May want to benchmark on additional datasets
    - https://www.plant-phenotyping.org/datasets-download
    - https://www.kaggle.com/c/data-science-bowl-2018
TODO: May want to consider some tweaks introduced in newer papers
    - https://arxiv.org/abs/1703.10277
TODO: Implement pixel clustering to generate final instance segmentation
    - https://gitlab.com/jhring/DataScienceBowl2018/-/blob/develop/src/discriminative_viz.py
"""
from vailtools.data.generate import SticksSequence
from vailtools.losses import discriminative_loss
from vailtools.networks import res_u_net


def main(batch_size=32, batches_per_epoch=10, epochs=5):
    dataset = SticksSequence(batch_size=batch_size, batches_per_epoch=batches_per_epoch)
    x_batch, y_batch = dataset[0]
    print(f"Train Input Shape: {x_batch.shape}")
    print(f"Train Label Shape: {y_batch.shape}")

    print("Building model...")
    model = res_u_net(
        depth=3,
        filters=16,
        coord_features=True,
        final_activation="linear",
        input_shape=x_batch.shape[1:],
        noise_std=0.003,
        output_channels=3,
    )
    model.compile(
        loss=discriminative_loss,
        optimizer="adam",
    )
    model.summary()

    # Ignore the warnings regarding interactions between Tensorflow and multiprocessing
    # Our dataset generator does not depend on any external resources, so deadlocks are unlikely
    model.fit(dataset, epochs=epochs, workers=4, use_multiprocessing=True)
    results = model.evaluate(dataset)
    print(f"Test loss: {results}")


if __name__ == "__main__":
    main()
