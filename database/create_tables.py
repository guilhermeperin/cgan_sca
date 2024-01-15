from sqlalchemy import Integer, String, ARRAY, Float, JSON, Boolean, DateTime, ForeignKey
from datetime import datetime


def create_tables(db):
    """Create 'ATTACKS' table."""
    attack_columns = [
        {"name": "reference", "type": String},
        {"name": "target", "type": String},
        {"name": "leakage_model", "type": String},
        {"name": "reference_key_byte", "type": Integer},
        {"name": "target_key_byte", "type": Integer},
        {"name": "cgan_features", "type": Integer},
        {"name": "datetime", "type": DateTime, "properties": {"default": datetime.utcnow}},
    ]
    db.create_table("attacks", attack_columns)

    """Create 'GENERATOR' table."""
    generator_columns = [
        {"name": "layers", "type": Integer},
        {"name": "neurons_1", "type": Integer},
        {"name": "neurons_2", "type": Integer},
        {"name": "neurons_3", "type": Integer},
        {"name": "neurons_4", "type": Integer},
        {"name": "neurons_5", "type": Integer},
        {"name": "neurons_6", "type": Integer},
        {"name": "activation", "type": String},
        {"name": "attack_id", "type": Integer, "foreign_key_table": "attacks"},
    ]
    db.create_table("generators", generator_columns)

    """Create 'DISCRIMINATOR' table."""
    discriminator_columns = [
        {"name": "layers_embedding", "type": Integer},
        {"name": "neurons_embedding", "type": Integer},
        {"name": "layers_dropout", "type": Integer},
        {"name": "neurons_dropout", "type": Integer},
        {"name": "dropout", "type": Float},
        {"name": "activation", "type": String},
        {"name": "attack_id", "type": Integer, "foreign_key_table": "attacks"},
    ]
    db.create_table("discriminators", discriminator_columns)

    """Create 'TRAININGS' table."""
    trainings_columns = [
        {"name": "batch_size", "type": Integer},
        {"name": "epochs", "type": Integer},
        {"name": "attack_id", "type": Integer, "foreign_key_table": "attacks"},
    ]
    db.create_table("trainings", trainings_columns)

    """Create 'PROFILINGS' table."""
    profilings_columns = [
        {"name": "batch_size", "type": Integer},
        {"name": "epochs", "type": Integer},
        {"name": "layers", "type": Integer},
        {"name": "neurons", "type": Integer},
        {"name": "activation", "type": String},
        {"name": "attack_id", "type": Integer, "foreign_key_table": "attacks"},
    ]
    db.create_table("profilings", profilings_columns)

    """Create 'PROFILINGS' table."""
    results_columns = [
        {"name": "accuracy_ref", "type": JSON},
        {"name": "accuracy_target", "type": JSON},
        {"name": "loss_generator", "type": JSON},
        {"name": "loss_discriminator", "type": JSON},
        {"name": "snr_ref", "type": JSON},
        {"name": "snr_target_real", "type": JSON},
        {"name": "snr_target", "type": JSON},
        {"name": "snr_share1", "type": JSON},
        {"name": "snr_share2", "type": JSON},
        {"name": "ge_traces", "type": JSON},
        {"name": "ge_epochs", "type": JSON},
        {"name": "nt_epochs", "type": JSON},
        {"name": "pi_epochs", "type": JSON},
        {"name": "attack_epochs_interval", "type": Integer},
        {"name": "attack_id", "type": Integer, "foreign_key_table": "attacks"},
    ]
    db.create_table("results", results_columns)
