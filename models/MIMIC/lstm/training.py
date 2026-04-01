import csv
import torch
from torch.nn.utils.clip_grad import clip_grad_norm_
from tqdm.notebook import tqdm
import wandb


def train_function(
    model,
    optimizer,
    scheduler,
    epochs,
    loss_function,
    train_data_loader,
    val_data_loader,
    device,
    clip_value=2.0,
    csv_file=None,
    patience=3,
    min_delta=0.0,
    use_wandb=True,
    wandb_project="Project Name",  # <-- Update with your W&B project name
    wandb_run_name=None,
    wandb_config=None,
    log_every_steps=200,
    best_model_path="lstm_best_model.pth", 
):

    # --- W&B init ---
    run = None
    if use_wandb:
        run = wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            config=wandb_config or {},
            reinit=True,
        )

    best_val_loss = float("inf")
    no_improve_count = 0

    train_loss_per_epoch = []
    val_loss_per_epoch = []

    csv_writer = None
    if csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["Epoch", "Train Loss", "Val Loss"])

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("Training...")

        model.train()
        total_loss = 0.0

        for step, batch in enumerate(tqdm(train_data_loader, desc="Training")):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            lengths = batch["lengths"].to(device)

            optimizer.zero_grad(set_to_none=True)

            logits = model(input_ids, lengths)
            loss = loss_function(logits, labels)

            loss.backward()

            # Gradient clipping
            grad_norm = clip_grad_norm_(model.parameters(), clip_value)

            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            total_loss += loss.item()

            # --- Step logging ---
            if (step + 1) % log_every_steps == 0:
                lr = optimizer.param_groups[0]["lr"]
                print(
                    f"Step {step + 1}/{len(train_data_loader)}, "
                    f"Loss: {loss.item():.6f}, LR: {lr:.3e}"
                )

                if use_wandb:
                    wandb.log(
                        {
                            "train/loss_step": loss.item(),
                            "train/grad_norm": grad_norm.item(),
                            "train/lr": lr,
                            "epoch": epoch + 1,
                            "global_step": epoch * len(train_data_loader) + step + 1,
                        }
                    )

        avg_train_loss = total_loss / max(1, len(train_data_loader))
        train_loss_per_epoch.append(avg_train_loss)
        print(f"Average training loss: {avg_train_loss:.6f}")

        # ---------------- Validation ----------------
        print("Validation...")
        model.eval()
        total_val_loss = 0.0

        with torch.no_grad():
            for val_batch in tqdm(val_data_loader, desc="Validation"):
                val_input_ids = val_batch["input_ids"].to(device)
                val_labels = val_batch["labels"].to(device)
                val_lengths = val_batch["lengths"].to(device)

                val_logits = model(val_input_ids, val_lengths)
                val_loss = loss_function(val_logits, val_labels)

                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / max(1, len(val_data_loader))
        val_loss_per_epoch.append(avg_val_loss)
        print(f"Average validation loss: {avg_val_loss:.6f}")

        # --- Epoch-level logging ---
        if use_wandb:
            wandb.log(
                {
                    "train/loss_epoch": avg_train_loss,
                    "val/loss": avg_val_loss,
                    "epoch": epoch + 1,
                }
            )

        # ---------------- Early Stopping ----------------
        improved = (best_val_loss - avg_val_loss) > min_delta

        if improved:
            best_val_loss = avg_val_loss
            no_improve_count = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"Model improved; saved to {best_model_path}")

            if use_wandb:
                wandb.run.summary["best_val_loss"] = best_val_loss
                wandb.run.summary["best_epoch"] = epoch + 1
        else:
            no_improve_count += 1
            print(f"No improvement for {no_improve_count}/{patience} epoch(s).")

        if csv_writer:
            csv_writer.writerow([epoch + 1, avg_train_loss, avg_val_loss])

        if no_improve_count >= patience:
            print(f"Early stopping: no val-loss improvement for {patience} epochs.")
            break

    if use_wandb and run is not None:
        run.finish()

    return model