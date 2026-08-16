Let me fix the Mermaid syntax errors. Here are the corrected diagrams:

---

## 1. Overall Project Architecture

```mermaid
flowchart TB
    subgraph INPUTS["📥 Inputs"]
        SEED["Student ID 121314"]
        DATA["CIFAR-10 Dataset"]
        IMAGES["Sample Images Bonus"]
    end

    subgraph CORE["⚙️ Core Modules"]
        CONFIG["config.py<br/>Hyperparameters"]
        DATA_MOD["data.py<br/>Loading and Splits"]
        MODELS["models.py<br/>SmallCNN, ResNet18"]
        AUGMENT["augment.py<br/>Mixup, CutMix"]
        ENGINE["engine.py<br/>Training Loop"]
        UTILS["utils.py<br/>Metrics, Plotting"]
    end

    subgraph EXPERIMENTS["🧪 Experiments"]
        B1["B1: train_cnn.py<br/>SmallCNN from Scratch"]
        B2["B2: transfer_compare.py<br/>Transfer Learning"]
        B3["B3: augment_compare.py<br/>Augmentation Study"]
        BONUS["Bonus: bonus_inference.py<br/>Pretrained Models"]
    end

    subgraph OUTPUTS["📤 Outputs"]
        FIGURES["figures/*.png<br/>Training Curves"]
        MODELS_OUT["models/*.pth<br/>Checkpoints"]
        HISTORIES["models/*.csv<br/>Training Histories"]
        ZOO["figures/zoo_*.png<br/>Annotated Images"]
    end

    SEED --> CONFIG
    DATA --> DATA_MOD
    IMAGES --> BONUS

    CONFIG --> B1
    CONFIG --> B2
    CONFIG --> B3
    CONFIG --> BONUS

    DATA_MOD --> B1
    DATA_MOD --> B2
    DATA_MOD --> B3

    MODELS --> B1
    MODELS --> B2
    MODELS --> B3

    AUGMENT --> B3
    ENGINE --> B1
    ENGINE --> B2
    ENGINE --> B3
    UTILS --> B1
    UTILS --> B2
    UTILS --> B3

    B1 --> FIGURES
    B1 --> MODELS_OUT
    B1 --> HISTORIES
    B2 --> FIGURES
    B2 --> MODELS_OUT
    B2 --> HISTORIES
    B3 --> FIGURES
    B3 --> MODELS_OUT
    B3 --> HISTORIES
    BONUS --> ZOO
```

---

## 2. Data Pipeline Flow

```mermaid
flowchart TD
    START["Start: get_cifar10_subset()"]
    START --> CHECK_CACHE{"Image Size = 224?"}

    CHECK_CACHE -->|"Yes (ResNet)"| CACHE_CHECK{"Cache exists?"}
    CACHE_CHECK -->|"Yes"| LOAD_CACHE["Load from data/cache/*.pkl"]
    CACHE_CHECK -->|"No"| CREATE_CACHE["Resize to 224x224<br/>Apply transforms<br/>Save to cache"]

    CHECK_CACHE -->|"No (SmallCNN)"| LOAD_DIRECT["Load directly<br/>with CIFAR-10 transforms"]

    LOAD_CACHE --> NORM["Apply normalization"]
    CREATE_CACHE --> NORM
    LOAD_DIRECT --> FILTER["Filter to 5 classes"]

    NORM --> FILTER
    FILTER --> SPLIT["Stratified split<br/>640 train / 160 val / 200 test<br/>per class (seed controlled)"]

    SPLIT --> DATASETS["Return: train, val, test"]
    DATASETS --> LOADERS["make_loaders()<br/>Wrap in DataLoaders<br/>batch_size=64"]
    LOADERS --> DATA_LOADERS["DataLoaders: train, val, test"]
```

---

## 3. B1: SmallCNN Training

```mermaid
flowchart TD
    START["Start: train_cnn.py"]
    START --> SEED["set_seed(121314)"]
    SEED --> DATA["Load CIFAR-10 subset<br/>32x32, CIFAR-10 norm"]
    DATA --> MODEL["Create SmallCNN<br/>3 conv blocks + GAP<br/>94,341 params"]
    MODEL --> OPTIM["Adam optimizer<br/>lr = 1e-3"]
    OPTIM --> LOOP["Training Loop<br/>20 epochs"]

    LOOP --> EPOCH["For each epoch"]
    EPOCH --> TRAIN["train_one_epoch()"]
    TRAIN --> EVAL["evaluate() on val"]
    EVAL --> LOG["Log: loss and accuracy"]
    LOG --> EPOCH

    LOOP --> SAVE_MODEL["Save model<br/>models/smallcnn.pth"]
    LOOP --> SAVE_HIST["Save history<br/>models/smallcnn_history.csv"]
    LOOP --> PLOT["plot_history()<br/>figures/cnn_curves.png"]

    SAVE_MODEL --> TEST["evaluate() on test"]
    TEST --> RESULT["Test Accuracy: 70.30%"]
```

---

## 4. B2: Transfer Learning Comparison

```mermaid
flowchart TD
    START["Start: transfer_compare.py"]
    START --> SEED["set_seed(121314)"]
    SEED --> DATA["Load CIFAR-10 subset<br/>224x224, ImageNet norm"]

    DATA --> FE["Feature Extraction Mode"]
    DATA --> FT["Fine-Tuning Mode"]

    FE --> FE_MODEL["build_resnet18('feature_extract')"]
    FE_MODEL --> FE_FREEZE["Freeze backbone<br/>requires_grad=False"]
    FE_FREEZE --> FE_HEAD["New fc layer<br/>Linear(512 to 5)"]
    FE_HEAD --> FE_TRAIN["Train 20 epochs<br/>lr = 1e-3"]
    FE_TRAIN --> FE_SAVE["Save: feature_extract_model.pth<br/>feature_extract_history.csv"]
    FE_SAVE --> FE_RESULT["Test Acc: 86.40%<br/>Params: 2,565"]

    FT --> FT_MODEL["build_resnet18('finetune')"]
    FT_MODEL --> FT_TRAINABLE["All layers trainable"]
    FT_TRAINABLE --> FT_HEAD["New fc layer<br/>Linear(512 to 5)"]
    FT_HEAD --> FT_TRAIN["Train 20 epochs<br/>lr = 1e-4"]
    FT_TRAIN --> FT_SAVE["Save: finetune_model.pth<br/>finetune_history.csv<br/>best_resnet_ft.pth"]
    FT_SAVE --> FT_RESULT["Test Acc: 93.70%<br/>Params: 11,179,077"]

    FE_RESULT --> COMPARE["plot_transfer_comparison()"]
    FT_RESULT --> COMPARE
    CNN_HIST["Load SmallCNN history"] --> COMPARE

    COMPARE --> PLOT["figures/transfer_compare.png"]
```

---

## 5. B3: Augmentation Study

```mermaid
flowchart TD
    START["Start: augment_compare.py"]
    START --> SEED["set_seed(121314)"]
    SEED --> LOAD_MODEL["Load best_resnet_ft.pth<br/>from B2"]

    LOAD_MODEL --> REGIME1["Regime 1: None"]
    LOAD_MODEL --> REGIME2["Regime 2: Standard"]
    LOAD_MODEL --> REGIME3["Regime 3: Mixup"]

    REGIME1 --> R1_DATA["Base dataset<br/>No additional transforms"]
    R1_DATA --> R1_TRAIN["Train 20 epochs"]
    R1_TRAIN --> R1_SAVE["Save: augment_none.pth<br/>none_history.csv"]
    R1_SAVE --> R1_RESULT["Test Acc: 94.00%"]

    REGIME2 --> R2_TRANSFORM["RandomCrop(224, padding=8)<br/>+ RandomHorizontalFlip"]
    R2_TRANSFORM --> R2_DATA["Augmented dataset"]
    R2_DATA --> R2_TRAIN["Train 20 epochs"]
    R2_TRAIN --> R2_SAVE["Save: augment_std.pth<br/>std_history.csv"]
    R2_SAVE --> R2_RESULT["Test Acc: 91.70%"]

    REGIME3 --> R3_TRANSFORM["lambda ~ Beta(1,1)<br/>x_mixed = lambda*x1 + (1-lambda)*x2"]
    R3_TRANSFORM --> R3_TRAIN["Train 20 epochs<br/>mix='mixup'"]
    R3_TRAIN --> R3_SAVE["Save: augment_mix.pth<br/>mix_history.csv"]
    R3_SAVE --> R3_RESULT["Test Acc: 93.40%"]

    R1_RESULT --> COMPARE["plot_augment_comparison()"]
    R2_RESULT --> COMPARE
    R3_RESULT --> COMPARE

    COMPARE --> PLOT["figures/augment_compare.png"]
```

---

## 6. Bonus: Pretrained Model Inference

```mermaid
flowchart TD
    START["Start: bonus_inference.py"]
    START --> SCAN["Scan data/samples/<br/>for images (JPG, PNG)"]
    SCAN --> CHECK{"Images found?"}

    CHECK -->|"No"| EXIT["Exit: No images"]
    CHECK -->|"Yes"| LOOP["For each image (max 5)"]

    LOOP --> DETECTION["Detection<br/>fasterrcnn_resnet50_fpn"]
    LOOP --> SEGMENTATION["Segmentation<br/>maskrcnn_resnet50_fpn"]
    LOOP --> POSE["Pose<br/>keypointrcnn_resnet50_fpn"]

    DETECTION --> D_MODEL["Load model"]
    D_MODEL --> D_INFER["Inference"]
    D_INFER --> D_BOXES["Draw boxes with labels/scores"]
    D_BOXES --> D_SAVE["Save: zoo_detection_*.png"]

    SEGMENTATION --> S_MODEL["Load model"]
    S_MODEL --> S_INFER["Inference"]
    S_INFER --> S_MASKS["Overlay instance masks"]
    S_MASKS --> S_SAVE["Save: zoo_segmentation_*.png"]

    POSE --> P_MODEL["Load model"]
    P_MODEL --> P_INFER["Inference"]
    P_INFER --> P_SKEL["Draw keypoints and skeleton"]
    P_SKEL --> P_SAVE["Save: zoo_pose_*.png"]

    D_SAVE --> DONE["All outputs saved"]
    S_SAVE --> DONE
    P_SAVE --> DONE

    DONE --> COMPLETE["Bonus Complete!"]
```

---

## 7. run_all.py Orchestration

```mermaid
flowchart TD
    START["Start: run_all.py"]
    START --> PARSE["Parse arguments<br/>--verbose, --quiet, --debug"]
    PARSE --> SETUP["Setup logging"]
    SETUP --> SEED["set_seed(121314)"]
    SEED --> DIRS["Create directories<br/>data/, models/, figures/"]

    DIRS --> B1["B1: train_cnn.py"]
    B1 --> B1_CHECK{"B1 success?"}
    B1_CHECK -->|"No"| ERROR1["Exit with error"]
    B1_CHECK -->|"Yes"| B2["B2: transfer_compare.py"]

    B2 --> B2_CHECK{"B2 success?"}
    B2_CHECK -->|"No"| ERROR2["Exit with error"]
    B2_CHECK -->|"Yes"| B3["B3: augment_compare.py"]

    B3 --> B3_CHECK{"B3 success?"}
    B3_CHECK -->|"No"| ERROR3["Exit with error"]
    B3_CHECK -->|"Yes"| BONUS["Bonus: bonus_inference.py"]

    BONUS --> BONUS_CHECK{"Bonus success?"}
    BONUS_CHECK -->|"No"| BONUS_WARN["Log warning<br/>(bonus optional)"]
    BONUS_CHECK -->|"Yes"| DONE

    BONUS_WARN --> DONE["All complete!"]
    DONE --> SUMMARY["Log generated figures"]
```

---

## 8. Smart Fallback Logic

```mermaid
flowchart TD
    START["Function: get_regime_history(regime_name)"]
    START --> CHECK{"history.csv exists<br/>AND .pth exists?"}

    CHECK -->|"Yes"| LOAD["Load from cache: history = load_history()<br/>model_path = existing .pth"]
    LOAD --> DONE["Return history, model_path<br/>(no training!)"]

    CHECK -->|"No"| TRAIN["Train from scratch"]
    TRAIN --> SAVE_HIST["save_history(history)"]
    SAVE_HIST --> SAVE_MODEL["torch.save(model.state_dict())"]
    SAVE_MODEL --> DONE2["Return history, model_path<br/>(saved for next time)"]
```

---

## 9. Complete Data Flow

```mermaid
flowchart LR
    subgraph DATA_FLOW["Data Flow"]
        CIFAR["CIFAR-10"] --> SPLIT["Split (seed 121314)<br/>640/160/200 per class"]
        SPLIT --> TRAIN_DS["Train DS 3,200"]
        SPLIT --> VAL_DS["Val DS 800"]
        SPLIT --> TEST_DS["Test DS 1,000"]

        TRAIN_DS --> B1_TRAIN["B1 Train"]
        TRAIN_DS --> B2_TRAIN["B2 Train"]
        TRAIN_DS --> B3_TRAIN["B3 Train"]

        VAL_DS --> B1_VAL["B1 Val"]
        VAL_DS --> B2_VAL["B2 Val"]
        VAL_DS --> B3_VAL["B3 Val"]

        TEST_DS --> B1_TEST["B1 Test"]
        TEST_DS --> B2_TEST["B2 Test"]
        TEST_DS --> B3_TEST["B3 Test"]
    end

    subgraph MODEL_FLOW["Model Flow"]
        SMALLCNN["SmallCNN"] --> B1_TRAIN
        RESNET["ResNet-18"] --> B2_TRAIN
        BEST["best_resnet_ft.pth"] --> B3_TRAIN
    end

    subgraph OUTPUT_FLOW["Output Flow"]
        B1_TRAIN --> CNN_CURVES["cnn_curves.png"]
        B2_TRAIN --> TRANSFER["transfer_compare.png"]
        B3_TRAIN --> AUGMENT["augment_compare.png"]

        B1_TRAIN --> SMALLCNN_PTH["smallcnn.pth"]
        B2_TRAIN --> FE_PTH["feature_extract_model.pth"]
        B2_TRAIN --> FT_PTH["finetune_model.pth"]
        B2_TRAIN --> BEST_PTH["best_resnet_ft.pth"]
        B3_TRAIN --> AUG_PTH["augment_*.pth"]

        B1_TRAIN --> HIST_CSV["*_history.csv"]
        B2_TRAIN --> HIST_CSV
        B3_TRAIN --> HIST_CSV
    end
```

---

## 10. Test Workflow

```mermaid
flowchart TD
    START["Run: uv run pytest tests/ -v"]
    START --> COLLECT["Collect 70 tests"]

    COLLECT --> DATA["test_data.py<br/>10 tests"]
    COLLECT --> MODELS["test_models.py<br/>15 tests"]
    COLLECT --> AUG["test_augment.py<br/>16 tests"]
    COLLECT --> ENGINE["test_engine.py<br/>14 tests"]
    COLLECT --> UTILS["test_utils.py<br/>15 tests"]

    DATA --> DATA_RES["✅ 10/10"]
    MODELS --> MODELS_RES["✅ 15/15"]
    AUG --> AUG_RES["✅ 16/16"]
    ENGINE --> ENGINE_RES["✅ 14/14"]
    UTILS --> UTILS_RES["✅ 15/15"]

    DATA_RES --> COVERAGE["Coverage: 77%"]
    MODELS_RES --> COVERAGE
    AUG_RES --> COVERAGE
    ENGINE_RES --> COVERAGE
    UTILS_RES --> COVERAGE

    COVERAGE --> DONE["✅ All tests pass!"]
```

---

## 11. File Dependencies

```mermaid
flowchart TD
    subgraph CORE["Core Modules"]
        CONFIG["config.py"]
        DATA["data.py"]
        MODELS["models.py"]
        AUG["augment.py"]
        ENGINE["engine.py"]
        UTILS["utils.py"]
    end

    subgraph EXPS["Experiments"]
        B1["train_cnn.py"]
        B2["transfer_compare.py"]
        B3["augment_compare.py"]
        BONUS["bonus_inference.py"]
    end

    subgraph TESTS["Tests"]
        TDATA["test_data.py"]
        TMODELS["test_models.py"]
        TAUG["test_augment.py"]
        TENG["test_engine.py"]
        TUTILS["test_utils.py"]
    end

    B1 --> MODELS
    B1 --> ENGINE
    B1 --> UTILS
    B1 --> CONFIG

    B2 --> MODELS
    B2 --> ENGINE
    B2 --> UTILS
    B2 --> CONFIG
    B2 --> B1_HIST["(loads SmallCNN history)"]

    B3 --> MODELS
    B3 --> ENGINE
    B3 --> UTILS
    B3 --> CONFIG
    B3 --> B2_MODEL["(loads best_resnet_ft.pth)"]

    BONUS --> CONFIG

    TDATA --> DATA
    TDATA --> CONFIG

    TMODELS --> MODELS
    TMODELS --> CONFIG

    TAUG --> AUG

    TENG --> ENGINE
    TENG --> MODELS
    TENG --> CONFIG

    TUTILS --> UTILS
```

---

## 12. Training Loop Sequence

```mermaid
sequenceDiagram
    participant User
    participant run_all
    participant Experiment
    participant Engine
    participant Model
    participant DataLoader

    User->>run_all: python run_all.py
    run_all->>run_all: set_seed(121314)

    loop For each experiment (B1, B2, B3)
        run_all->>Experiment: Run experiment script
        Experiment->>DataLoader: Load data
        DataLoader-->>Experiment: DataLoaders

        Experiment->>Model: Create model
        Experiment->>Engine: fit()

        loop For each epoch (1 to 20)
            Engine->>Model: train_one_epoch()
            Model->>DataLoader: Get batch
            DataLoader-->>Model: (x, y)
            Model->>Model: forward()
            Model->>Model: backward()
            Model->>Model: optimizer.step()
            Model-->>Engine: loss, acc

            Engine->>Model: evaluate()
            Model->>DataLoader: Get val batch
            DataLoader-->>Model: (x_val, y_val)
            Model->>Model: forward()
            Model-->>Engine: val_loss, val_acc

            Engine-->>Engine: Log progress
        end

        Engine-->>Experiment: history
        Experiment->>Experiment: Save model
        Experiment->>Experiment: Save history
        Experiment->>Experiment: Plot curves
        Experiment-->>run_all: Complete
    end

    run_all-->>User: All experiments complete!
```

---

## Summary Table

| Phase | Component | Input | Output | Time |
|-------|-----------|-------|--------|------|
| **Data** | `data.py` | CIFAR-10 | 3,200/800/1,000 split | ~2-3 min |
| **B1** | `train_cnn.py` | Data, SmallCNN | `cnn_curves.png`, `smallcnn.pth` | ~10-15 min |
| **B2** | `transfer_compare.py` | Data, ResNet-18 | `transfer_compare.png`, checkpoints | ~20-30 min |
| **B3** | `augment_compare.py` | Data, best model | `augment_compare.png`, 3 models | ~30-45 min |
| **Bonus** | `bonus_inference.py` | Sample images | `zoo_*.png` | ~10-20 sec |
| **Tests** | `pytest` | All src/ | 70 passing tests | ~3-4 min |
| **All** | `run_all.py` | Everything | All outputs | ~1-1.5 hours |