# Qlib Module Learning Plan

This document is a hands-on plan for learning how this Qlib repository works. The goal is not merely to learn how to run a benchmark. By the end, you should be able to trace a value from a raw market-data file through feature computation, preprocessing, model training, signal generation, portfolio construction, simulated execution, evaluation, and experiment storage. You should also be able to replace each major component with your own implementation.

The plan is organized around the following pipeline:

```text
qlib.init(configuration)
        |
        v
data storage -> providers -> expression engine -> D API
        |
        v
data loader -> data handler -> processors -> DatasetH
        |
        v
model.fit(dataset) -> model.predict(dataset) -> prediction scores
        |
        v
strategy -> trade decisions/orders -> executor -> exchange/account
        |
        v
backtest metrics + signal analysis + saved experiment artifacts
```

## 1. Target outcomes

After completing the core modules, you should be able to answer these questions without guessing:

1. What global state does `qlib.init()` create, and why must initialization happen before data access?
2. How does `D.features()` turn expressions such as `$close` and `Ref($close, 1)` into a DataFrame?
3. What is the difference between a provider, loader, handler, processor, and dataset?
4. Why does `DataHandlerLP` maintain raw, learning, and inference views of data?
5. What contract must a Qlib model implement?
6. How does a model prediction become a strategy signal and then a set of orders?
7. What responsibilities belong to a strategy, executor, exchange, account, and position?
8. How does Qlib avoid coupling model code directly to backtest code?
9. What does `R` record, and how do record templates connect dependent workflow stages?
10. How does a YAML configuration become live Python objects?
11. Where should a new data processor, model, strategy, or report be added?
12. Which parts are stable framework abstractions and which parts are contributed implementations?

The advanced modules should additionally let you explain rolling training, online updates, nested execution, high-frequency workflows, meta-learning, and reinforcement learning.

## 2. How to use this plan

For every module, use the same four-pass method:

1. **Map:** identify the public classes, their responsibilities, and their dependencies.
2. **Trace:** follow one concrete input through the relevant call chain.
3. **Run:** execute or adapt a small example and inspect its intermediate objects.
4. **Rebuild:** implement a deliberately small version of the abstraction yourself.

Keep notes under `self_learn/`. A useful structure is:

```text
self_learn/
├── module_learning.md       # this roadmap and progress checklist
├── notes/                   # conceptual notes, one file per module
├── experiments/             # small runnable scripts
├── configs/                 # modified workflow YAML files
└── diagrams/                # call graphs and data-flow diagrams
```

For each module, write a short learning record containing:

```markdown
## What I expected

## What the code actually does

## Important classes and contracts

## One traced call path

## Experiment and result

## Questions still open
```

Do not try to memorize every implementation. Focus on boundaries: what an object receives, what it returns, who constructs it, and who calls it.

## 3. Prerequisites and environment check

Before studying internals, be comfortable with:

- Python classes, inheritance, abstract base classes, generators, and context managers
- NumPy and pandas, especially `DataFrame`, `Series`, `MultiIndex`, and time slicing
- Basic supervised learning: features, labels, training, validation, prediction, and overfitting
- Basic portfolio concepts: holdings, cash, benchmark, return, turnover, transaction cost, and drawdown
- YAML syntax

Inspect the environment without changing the source:

```bash
python --version
python -c "import qlib; print(qlib.__version__)"
python -c "import pandas, numpy, lightgbm, mlflow; print('dependencies import correctly')"
qrun --help
```

If the repository is installed in editable mode, confirm which source is imported:

```bash
python -c "import qlib; print(qlib.__file__)"
```

Expected result: the printed path should point into this repository's `qlib/` directory.

---

# Part I — Orientation and object construction

## Module 0: Build the package map

**Goal:** distinguish framework code, contributed implementations, examples, tools, documentation, and tests.

### Read

1. `README.md`, especially “Framework of Qlib” and the quick-start workflow
2. `docs/introduction/introduction.rst`
3. `pyproject.toml`
4. Top-level directories under `qlib/`, `examples/`, `scripts/`, and `tests/`

### Learn the repository roles

| Location | Role |
|---|---|
| `qlib/data` | Data storage, retrieval, expressions, caching, and dataset preparation |
| `qlib/model` | Core model contracts, training helpers, ensembles, interpretation, and risk models |
| `qlib/strategy` | Core strategy interfaces |
| `qlib/backtest` | Decisions, execution, market simulation, positions, accounts, and metrics |
| `qlib/workflow` | Experiments, recorders, record templates, tasks, and online workflows |
| `qlib/rl` | Reinforcement-learning abstractions and order-execution applications |
| `qlib/contrib` | Ready-to-use models, handlers, strategies, reports, and extensions |
| `qlib/utils` | Object construction, serialization, time, parallelism, and shared utilities |
| `examples` | End-to-end usage and benchmark configurations |
| `scripts` | Data collection, conversion, checking, and operational tools |
| `tests` | Executable specifications of expected behavior |

### Exercise

Draw a one-page dependency map. Put `qlib/data` at the beginning, `qlib/model` and `qlib/strategy` in the middle, and `qlib/workflow` around the whole lifecycle. Mark `qlib/contrib` as implementations built on core interfaces.

### Checkpoint

Explain why `qlib/contrib/model/gbdt.py` is not the model abstraction and why `qlib/model/base.py` is not a usable forecasting algorithm by itself.

### Learning artifact

Create `self_learn/notes/00_package_map.md` with your map and a one-sentence responsibility for every top-level package.

## Module 1: Initialization, configuration, and dynamic object creation

**Goal:** understand how configuration controls the rest of Qlib and how YAML descriptions become objects.

### Read in order

1. `qlib/__init__.py`: `init()` and `init_from_yaml_conf()`
2. `qlib/config.py`: `QlibConfig`, defaults, registration, provider configuration, and region settings
3. `qlib/utils/__init__.py`: find `init_instance_by_config`
4. `qlib/cli/run.py`: follow the `qrun` entry point
5. `examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml`
6. `docs/component/workflow.rst`

### Concepts to understand

- Qlib has global configuration, exposed conventionally as `C`.
- `qlib.init()` selects and registers storage/data providers and experiment settings.
- Many objects are described using dictionaries containing `class`, `module_path`, and `kwargs`.
- `init_instance_by_config` imports and constructs those objects dynamically.
- Placeholders such as `<MODEL>`, `<DATASET>`, and `<PRED>` connect workflow stages.
- `qrun` is an orchestration layer; it does not implement modeling or backtesting itself.

### Trace exercise

Starting at the YAML file, trace these fields into Python:

```text
qlib_init
task.model
task.dataset
task.record
```

For each field, write down:

- Which function reads it
- Which Python class it creates
- Which keyword arguments reach the constructor
- When the resulting object is used

### Coding exercise

Write `self_learn/experiments/01_config_construction.py` that calls `init_instance_by_config` for a tiny class configuration. First use a Qlib class, then define your own local class and construct it with the same mechanism.

### Checkpoint

Given this fragment, explain exactly what is imported and constructed:

```yaml
model:
  class: LGBModel
  module_path: qlib.contrib.model.gbdt
  kwargs:
    loss: mse
```

---

# Part II — Data infrastructure

## Module 2: Physical data layout and storage abstractions

**Goal:** understand what Qlib data contains on disk and how storage classes expose it.

### Read in order

1. `docs/component/data.rst`, through data preparation and conversion
2. `docs/start/getdata.rst`
3. `qlib/data/storage/storage.py`
4. `qlib/data/storage/file_storage.py`
5. `scripts/dump_bin.py`
6. `scripts/get_data.py`

### Concepts to understand

- Calendars define valid timestamps.
- Instruments define markets/universes and membership spans.
- Feature storage holds time-indexed fields for instruments.
- Qlib's binary data layout is optimized for repeated quantitative queries.
- Storage interfaces separate logical access from physical file layout.
- Raw CSV or Parquet data generally needs validation and conversion before local providers can use it efficiently.

### Inspection exercise

If local Qlib data is available, inspect its directory structure without modifying it. Identify calendar files, instrument definitions, and feature files. Relate each file category to its corresponding storage class.

### Trace exercise

Choose one instrument and one field, such as `SH600000` and `close`. Trace how a request reaches `FileFeatureStorage` and how timestamps are mapped to stored values.

### Coding exercise

Create a tiny synthetic CSV dataset with two instruments and a few dates under `self_learn/experiments/data/`. Do not convert it immediately. First document what fields and ordering Qlib expects, then use the repository's supported conversion tooling to create a separate Qlib-format copy.

### Checkpoint

Explain why calendar and instrument membership are separate from feature values. Include an example involving a suspended stock or a stock entering an index.

## Module 3: Providers and the public `D` API

**Goal:** understand the facade through which users retrieve calendars, instruments, features, and expressions.

### Read in order

1. `qlib/data/__init__.py`
2. `qlib/data/data.py`: provider abstract classes
3. `qlib/data/data.py`: local provider implementations
4. `qlib/data/data.py`: `BaseProvider`, `LocalProvider`, and `ClientProvider`
5. `tests/test_get_data.py`
6. `tests/dataset_tests/test_datalayer.py`

### Main abstractions

| Abstraction | Responsibility |
|---|---|
| `CalendarProvider` | Return valid timestamps for a frequency |
| `InstrumentProvider` | Resolve a market or instrument specification |
| `FeatureProvider` | Read basic stored fields |
| `ExpressionProvider` | Evaluate feature expressions |
| `DatasetProvider` | Assemble multi-instrument, multi-expression data |
| `D` | Convenient facade delegating to configured providers |

### Run exercise

After initializing Qlib, inspect:

```python
from qlib.data import D

calendar = D.calendar(start_time="2020-01-01", end_time="2020-01-31")
universe = D.instruments("csi300")
data = D.features(
    universe,
    ["$close", "$volume"],
    start_time="2020-01-01",
    end_time="2020-01-31",
)
```

Record the Python type, index names, column names, shape, and missing-value behavior of each result.

### Trace exercise

Put temporary debugger breakpoints, or add a temporary local tracing script, around `D.features()`. Record the provider chain from the facade to storage. Avoid committing debug prints to Qlib source.

### Checkpoint

Explain the difference between selecting an instrument universe and loading its features. Why might the universe itself vary through time?

## Module 4: Expression engine, operators, and cache

**Goal:** learn how Qlib performs feature engineering from expression strings.

### Read in order

1. `qlib/data/base.py`: `Expression`, `Feature`, and `ExpressionOps`
2. `qlib/data/ops.py`
3. `qlib/data/data.py`: expression parsing and expression providers
4. `qlib/data/cache.py`
5. `tests/ops/test_elem_operator.py`
6. `tests/ops/test_special_ops.py`
7. `tests/test_register_ops.py`
8. `examples/data_demo/data_cache_demo.py`

### Expressions to study

Start with these and predict their output before running them:

```text
$close
Ref($close, 1)
$close / Ref($close, 1) - 1
Mean($close, 5)
Std($close, 20)
Corr($close, $volume, 10)
Rank($close)
```

Pay close attention to:

- Window boundaries
- Lookback requirements
- Missing values at the start of a series
- Per-instrument versus cross-sectional operations
- How an expression determines the extra data range it needs
- Where expression and dataset caches are applied

### Coding exercise

Implement one simple custom operator by following the operator registration mechanism. Test it on a short synthetic series where you can calculate the expected result by hand.

### Checkpoint

For `Mean(Ref($close, 1), 5)`, draw the expression tree and explain how much historical data is required to compute the first requested timestamp correctly.

---

# Part III — From raw features to model-ready datasets

## Module 5: Data loaders

**Goal:** understand how feature and label definitions become a raw tabular dataset.

### Read in order

1. `qlib/data/dataset/loader.py`
2. `qlib/contrib/data/loader.py`
3. `tests/data_mid_layer_tests/test_dataloader.py`
4. The loader configuration used by `Alpha158` in `qlib/contrib/data/handler.py`

### Concepts to understand

- `QlibDataLoader` retrieves expressions using the Qlib data API.
- `StaticDataLoader` accepts already-materialized data.
- Loader configuration groups columns, commonly into `feature` and `label`.
- Column groups become a column `MultiIndex`, while date and instrument normally form the row `MultiIndex`.
- A loader gathers data; it does not define train/validation/test segments.

### Coding exercise

Construct a `QlibDataLoader` with two features and one forward-return label. Load a short period and inspect every index level. Then construct a `StaticDataLoader` producing the same schema from a DataFrame.

### Checkpoint

Explain why a forward-return label is useful for training but must not be used as an inference feature.

## Module 6: Processors and `DataHandlerLP`

**Goal:** understand Qlib's preprocessing lifecycle and its protection against leakage.

### Read in order

1. `qlib/data/dataset/processor.py`
2. `qlib/data/dataset/handler.py`: `DataHandlerABC` and `DataHandler`
3. `qlib/data/dataset/handler.py`: `DataHandlerLP`
4. `qlib/contrib/data/processor.py`
5. `tests/data_mid_layer_tests/test_processor.py`
6. `tests/data_mid_layer_tests/test_handler.py`

### Core ideas

Learn the meanings of:

- `infer_processors`
- `learn_processors`
- Processor `fit()` versus processor `__call__()`
- Raw data view
- Inference data view
- Learning data view
- Independent versus append processing modes
- `fit_start_time` and `fit_end_time`

Use normalization as the main example. A normalization processor may estimate statistics from a training interval and later apply those fixed statistics to validation, test, or online data.

### Trace exercise

Choose one processor stack from `Alpha158`. For each processor, document:

1. Input columns
2. Whether it learns parameters
3. The period from which it learns them
4. Which data view it changes
5. Its output schema

### Coding exercise

Create a custom processor that clips a feature to learned training-set quantiles. Verify that changing test-period values does not change the fitted quantiles.

### Checkpoint

Explain a concrete leakage bug that would occur if normalization parameters were fitted on the full dataset.

## Module 7: `Dataset`, `DatasetH`, segments, and weights

**Goal:** understand the exact object passed into model training and prediction.

### Read in order

1. `qlib/data/dataset/__init__.py`
2. `qlib/data/dataset/weight.py`
3. `qlib/data/dataset/storage.py`
4. `tests/data_mid_layer_tests/test_dataset.py`
5. `tests/data_mid_layer_tests/test_handler_storage.py`

### Concepts to understand

- A dataset organizes prepared data for a learning task.
- Segments such as `train`, `valid`, and `test` are named time slices.
- `prepare()` selects segments, column groups, and data views.
- A reweighter can provide sample weights without changing features or labels.
- Handler storage and dataset preparation can trade memory for repeated-query speed.

### Run exercise

Given a `DatasetH`, compare:

```python
dataset.prepare("train", col_set="feature")
dataset.prepare("train", col_set=["feature", "label"])
dataset.prepare(["train", "valid"], col_set=["feature", "label"])
```

Record return types and schemas. Repeat using the learning and inference data keys where applicable.

### Rebuild exercise

Write a minimal custom `Dataset` implementation backed by a DataFrame. Support two named time segments and a `prepare()` method. The purpose is to understand the model-facing contract, not to reproduce all `DatasetH` features.

### Checkpoint

At this point, draw a complete data path from a `.bin` feature file to the `x_train` DataFrame received by a model.

---

# Part IV — Modeling and prediction records

## Module 8: Model contracts and training implementations

**Goal:** understand what Qlib requires from a model and how implementations use datasets.

### Read in order

1. `qlib/model/base.py`
2. `qlib/model/trainer.py`
3. `qlib/contrib/model/gbdt.py`
4. `qlib/contrib/model/linear.py`
5. One PyTorch implementation that you understand, such as `pytorch_lstm.py`
6. `tests/test_contrib_model.py`
7. `tests/model/test_general_nn.py`

### Compare three model styles

| Model | What to observe |
|---|---|
| Linear model | Simplest extraction of arrays and prediction path |
| LightGBM model | Training/validation datasets, early stopping, feature importance |
| PyTorch model | Batching, device handling, epochs, checkpointing, and reproducibility |

For every implementation, identify:

- How it calls `dataset.prepare()`
- Which data view it requests
- How it handles labels and optional weights
- Which learned attributes are stored on the model
- The index and type of `predict()` output
- How save/load behavior is supported

### Coding exercise

Implement a small baseline model under `self_learn/experiments/`, inheriting from Qlib's `Model`. It can predict a constant, a rolling mean, or use a small scikit-learn estimator if available. It must preserve the test sample index in its returned `Series`.

### Checkpoint

Explain why model predictions usually represent relative scores rather than literal expected returns, and why the strategy can still use them.

## Module 9: Experiment management and record templates

**Goal:** understand how Qlib makes research runs reproducible and connects workflow stages through artifacts.

### Read in order

1. `qlib/workflow/__init__.py`: `QlibRecorder` and global `R`
2. `qlib/workflow/exp.py`
3. `qlib/workflow/expm.py`
4. `qlib/workflow/recorder.py`
5. `qlib/workflow/record_temp.py`
6. `docs/component/recorder.rst`
7. `tests/test_workflow.py`

### Concepts to understand

- Experiment versus recorder/run
- Parameters, metrics, tags, and artifacts
- Context-managed run lifecycle
- Saving and loading Python objects
- MLflow-backed implementations behind Qlib interfaces
- Record-template dependencies
- Standard artifact names such as predictions and labels

### Trace exercise

Trace a workflow containing:

```text
SignalRecord -> SigAnaRecord -> PortAnaRecord
```

For each record template, list:

- Required constructor inputs
- Artifacts it reads
- Artifacts it writes
- Metrics it logs
- Which earlier record it depends on

### Coding exercise

Run a very small experiment and use `R` to save one parameter, one metric, and one Python object. End the process, reopen the recorder by ID, and load the object.

### Checkpoint

Explain why a `SignalRecord` belongs to the workflow layer rather than the model layer.

---

# Part V — Signals, portfolio decisions, and execution

## Module 10: From predictions to strategies

**Goal:** understand how model scores become portfolio or order decisions.

### Read in order

1. `qlib/backtest/signal.py`
2. `qlib/strategy/base.py`
3. `qlib/contrib/strategy/signal_strategy.py`
4. `qlib/contrib/strategy/order_generator.py`
5. `docs/component/strategy.rst`
6. `tests/backtest/test_soft_topk_strategy.py`

### Concepts to understand

- A signal provides time-indexed scores to a strategy.
- A strategy observes current state and produces a `TradeDecision`.
- Portfolio construction is distinct from prediction.
- `TopkDropoutStrategy` retains high-ranked holdings while controlling turnover through dropping and replacement rules.
- Weight-based strategies produce target portfolio weights and delegate order generation.
- Strategy state includes the current position, trading calendar, exchange, and prior execution results.

### Trace exercise

Take prediction scores for a single date and trace them through `TopkDropoutStrategy` into buy and sell orders. Record how these affect the result:

- `topk`
- `n_drop`
- Existing holdings
- Tradability
- Available cash
- Transaction costs

### Coding exercise

Implement a trivial strategy that holds the top three instruments with equal weight. First write it as pseudocode, then implement it by extending the closest Qlib base class.

### Checkpoint

Explain why comparing two models using different portfolio-construction rules would confound model quality with strategy quality.

## Module 11: Trade decisions and exchange simulation

**Goal:** understand the objects representing intended trades and the rules that determine whether they execute.

### Read in order

1. `qlib/backtest/decision.py`
2. `qlib/backtest/exchange.py`
3. `qlib/backtest/position.py`
4. `qlib/backtest/account.py`
5. `tests/backtest/test_file_strategy.py`

### Concepts to understand

- `Order` direction, amount, and execution state
- `TradeDecisionWO` and decision ranges
- Deal price
- Lot size/trade unit
- Price limits and suspended instruments
- Open cost, close cost, and minimum cost
- Cash constraints
- Position updates after execution
- Portfolio value versus cash balance

### Hand-calculation exercise

Construct a two-stock example with:

- Initial cash
- One buy and one sell order
- Explicit prices
- Open and close costs
- A minimum transaction fee

Calculate the expected cash, holdings, cost, and portfolio value by hand. Then reproduce it using the relevant Qlib objects and compare results.

### Checkpoint

Explain the difference between a trade decision, an order, and an execution result. Include an example where the decision exists but the order cannot be filled.

## Module 12: Executors and the backtest loop

**Goal:** understand simulated time, execution levels, and how strategies interact with executors.

### Read in order

1. `qlib/backtest/utils.py`: calendars and shared infrastructure
2. `qlib/backtest/executor.py`: `BaseExecutor`
3. `qlib/backtest/executor.py`: `SimulatorExecutor`
4. `qlib/backtest/executor.py`: `NestedExecutor`
5. `qlib/backtest/backtest.py`
6. `qlib/backtest/__init__.py`: construction helpers and public `backtest()`
7. `tests/backtest/test_high_freq_trading.py`

### Trace the core loop

Understand this sequence precisely:

```text
reset executor and calendar
while executor is not finished:
    strategy generates a decision
    executor executes or delegates the decision
    account and indicators are updated
    result returns to the strategy
    calendar advances
collect portfolio and execution metrics
```

### Generator exercise

The backtest collection path uses generators. Write a tiny independent generator example that mimics:

1. Yielding a decision
2. Receiving an externally supplied replacement decision
3. Returning a final result

Then reread `collect_data_loop()` and explain why a generator is useful for RL data collection and decision interception.

### Nested execution exercise

Draw the control flow for a daily portfolio decision executed by a minute-level strategy. Identify which calendar, strategy, executor, and exchange exist at each level.

### Checkpoint

Explain `CommonInfrastructure` versus `LevelInfrastructure`, and why nested execution needs both.

## Module 13: Metrics, evaluation, and reports

**Goal:** distinguish model/signal evaluation from portfolio/execution evaluation.

### Read in order

1. `qlib/contrib/eva/alpha.py`
2. `qlib/contrib/evaluate.py`
3. `qlib/contrib/evaluate_portfolio.py`
4. `qlib/backtest/report.py`
5. `qlib/workflow/record_temp.py`: `SigAnaRecord` and `PortAnaRecord`
6. `qlib/contrib/report/`
7. `docs/component/report.rst`

### Metrics to understand

**Signal/model layer:**

- Information coefficient (IC)
- Rank IC
- Long-short return
- Precision at selected ranks

**Portfolio layer:**

- Return and benchmark return
- Excess return
- Annualized return
- Volatility
- Sharpe ratio
- Maximum drawdown
- Turnover and transaction cost

**Execution layer:**

- Fill and trade indicators
- Price advantage/slippage-related measures
- Order-level statistics

### Analysis exercise

Create three synthetic prediction series:

1. Perfectly aligned with labels
2. Perfectly reversed
3. Random

Compute IC and rank IC. Then explain why strong IC does not guarantee a strong after-cost portfolio backtest.

### Checkpoint

Produce a table that maps each metric to the artifact/data required to calculate it and the Qlib function or record template that calculates it.

---

# Part VI — End-to-end workflow mastery

## Module 14: Trace one complete YAML workflow

**Goal:** connect every core component in a real run.

Use:

```text
examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml
```

### First pass: annotate configuration

Copy the configuration to `self_learn/configs/` and add comments explaining every field. Do not modify the original benchmark file.

### Second pass: write the object graph

Document the concrete instance created for every configured class:

```text
LGBModel
DatasetH
Alpha158 handler
processors
SignalRecord
SigAnaRecord
PortAnaRecord
TopkDropoutStrategy
executor and exchange
```

### Third pass: trace one sample

Choose one `(datetime, instrument)` pair and document:

1. Raw stored values used by its expressions
2. Computed Alpha158 feature values
3. Processor transformations
4. Dataset segment membership
5. Model prediction
6. Cross-sectional rank for that date
7. Strategy decision
8. Resulting order, if any
9. Execution price and cost
10. Position/account change
11. Metrics and saved artifacts affected

Some model internals aggregate many samples, so exact attribution may be difficult. When exact attribution is unavailable, document the boundary at which the sample enters and leaves the component.

### Fourth pass: run controlled variations

Change only one factor at a time in copied configurations:

1. Change model parameters but keep data and strategy fixed.
2. Change `topk` or `n_drop` but keep predictions fixed.
3. Change transaction costs but keep orders fixed.
4. Change train/validation/test boundaries.
5. Replace Alpha158 with a tiny custom handler.

Record which artifacts and metrics change. This demonstrates the separation of concerns experimentally.

### Final checkpoint

Explain the entire workflow aloud or in writing without referring to the framework diagram. Then verify your explanation against the call chain.

## Module 15: Rebuild a minimal workflow by code

**Goal:** remove the convenience of `qrun` and assemble the system yourself.

### Read

1. `examples/workflow_by_code.py`
2. `docs/component/model.rst`
3. `docs/component/workflow.rst`

### Build in stages

Write `self_learn/experiments/15_workflow_by_code.py` that explicitly:

1. Initializes Qlib.
2. Constructs a handler.
3. Constructs a dataset with three segments.
4. Constructs and fits a model.
5. Generates predictions.
6. Starts a recorder.
7. Saves the model and signal artifacts.
8. Calculates signal analysis.
9. Constructs a strategy and backtest configuration.
10. Runs a backtest.
11. Stores portfolio analysis.

Do not begin with every feature. Get steps 1–5 working first, then add recording, then backtesting.

### Compare YAML and code

Create a two-column note mapping every Python construction step to the equivalent YAML section. Identify what `qrun` automates and what remains the responsibility of the component implementations.

### Checkpoint

You have mastered the core architecture when you can debug a failed YAML run by identifying the corresponding Python object and reproducing its construction interactively.

---

# Part VII — Extension exercises

## Module 16: Add one component of each major kind

**Goal:** test architectural understanding through extension rather than passive reading.

Build these small components under `self_learn/experiments/` first:

1. **Expression/operator:** a simple derived feature.
2. **Processor:** a fitted clipping or normalization transformation.
3. **Data handler:** a handler with 3–5 features and one label.
4. **Model:** a baseline implementing `fit()` and `predict()`.
5. **Strategy:** an equal-weight top-N strategy.
6. **Record template:** a small report that reads predictions and saves one metric.

For each component, document:

- Base class or protocol
- Required methods
- Input/output schema
- Lifecycle and mutable state
- Serialization expectations
- How configuration constructs it
- Relevant tests to imitate

Finally, assemble these components into one custom workflow. Keep it small enough that intermediate results can be inspected manually.

## Module 17: Testing and debugging practices

**Goal:** learn the test suite as executable architecture documentation.

### Study these test areas

| Area | Tests |
|---|---|
| Expressions/operators | `tests/ops/` |
| Data providers | `tests/test_get_data.py`, `tests/dataset_tests/` |
| Loaders/handlers/datasets | `tests/data_mid_layer_tests/` |
| Models | `tests/test_contrib_model.py`, `tests/model/` |
| Workflow/recorder | `tests/test_workflow.py`, `tests/test_contrib_workflow.py` |
| Backtesting | `tests/backtest/` |
| Reinforcement learning | `tests/rl/` |

### Practices

- Run the smallest relevant test file before a broad suite.
- Use synthetic data when learning contracts.
- Assert index and column schemas, not just numeric values.
- Check time boundaries explicitly.
- Test for leakage by perturbing future data.
- Test transaction-cost and non-tradable edge cases.
- Use saved predictions to separate model debugging from strategy debugging.

### Exercise

For each custom component from Module 16, add at least:

1. A normal-case test
2. A boundary-case test
3. A schema/index assertion
4. A serialization or reconstruction test when relevant

---

# Part VIII — Advanced subsystems

Complete these only after Parts I–VII. Each subsystem reuses the core abstractions and will be much easier once the basic workflow is clear.

## Module 18: Rolling training and task management

**Goal:** understand repeated training as time moves forward.

### Read

1. `qlib/workflow/task/gen.py`
2. `qlib/workflow/task/manage.py`
3. `qlib/workflow/task/collect.py`
4. `qlib/contrib/rolling/base.py`
5. `examples/model_rolling/task_manager_rolling.py`
6. `tests/rolling_tests/`

### Questions

- How is a base task converted into multiple time-shifted tasks?
- Which windows train, validate, and predict?
- How are predictions combined across recorders?
- When is a model retrained versus reused or fine-tuned?
- How is task state tracked?

### Exercise

Draw three consecutive rolling tasks on a time axis, labeling training, validation, test, and prediction periods. Check the actual generated configurations against your drawing.

## Module 19: Online workflow

**Goal:** understand how models and predictions are updated as new data arrives.

### Read

1. `qlib/workflow/online/manager.py`
2. `qlib/workflow/online/strategy.py`
3. `qlib/workflow/online/update.py`
4. `qlib/workflow/online/utils.py`
5. `examples/online_srv/`
6. `docs/component/online.rst`

### Questions

- What is an online strategy in this context?
- How are new recorders prepared and promoted?
- How are existing prediction and label artifacts extended?
- What assumptions are made about newly available data?
- How does online management differ from live brokerage execution?

### Exercise

Simulate two update dates. Record which model, recorder, prediction range, and artifact version is active at each date.

## Module 20: Nested and high-frequency execution

**Goal:** understand multiple decision frequencies working together.

### Read

1. `docs/component/highfreq.rst`
2. `examples/nested_decision_execution/workflow.py`
3. `examples/highfreq/workflow.py`
4. `qlib/backtest/executor.py`: nested execution paths
5. `qlib/contrib/data/highfreq_handler.py`
6. `qlib/contrib/data/highfreq_processor.py`
7. `tests/backtest/test_high_freq_trading.py`

### Questions

- How is an outer decision constrained by its time range?
- How can an inner strategy split an outer order?
- What information is shared across levels?
- When can an outer strategy revise a decision?
- How are metrics reported at different frequencies?

### Exercise

Trace one daily order through minute-level slices. Record every decision and execution boundary.

## Module 21: Reinforcement learning

**Goal:** map RL concepts onto Qlib's strategy and execution abstractions.

### Read

1. `docs/component/rl/overall.rst`
2. `docs/component/rl/framework.rst`
3. `qlib/rl/simulator.py`
4. `qlib/rl/interpreter.py`
5. `qlib/rl/reward.py`
6. `qlib/rl/trainer/`
7. `qlib/rl/order_execution/`
8. `tests/rl/`

### Map the concepts

| RL concept | Qlib responsibility |
|---|---|
| Environment/simulator | Advances market/execution state |
| Observation | Extracted and interpreted state |
| Action | Interpreted into a trading decision |
| Reward | Evaluates an execution or portfolio outcome |
| Policy | Chooses actions |
| Trainer | Collects experience and updates policy |

### Exercise

Trace one environment step from observation construction through action interpretation, simulated execution, next state, and reward.

## Module 22: Meta-learning and market adaptation

**Goal:** understand how Qlib represents tasks as data for another learning layer.

### Read

1. `docs/component/meta.rst`
2. `qlib/model/meta/task.py`
3. `qlib/model/meta/dataset.py`
4. `qlib/model/meta/model.py`
5. `qlib/contrib/meta/data_selection/`
6. `qlib/contrib/rolling/ddgda.py`

### Questions

- What is stored in a meta-task?
- How does a meta-dataset organize tasks?
- Does the meta-model modify a base task or participate in base-model training?
- How does data selection adapt to market regimes?
- How is this different from ordinary rolling retraining?

### Exercise

Draw both levels explicitly: the base forecasting tasks and the meta-model that consumes information about those tasks.

## Module 23: Risk models and portfolio optimization

**Goal:** understand components that estimate risk or convert forecasts into constrained portfolios.

### Read

1. `qlib/model/riskmodel/base.py`
2. `qlib/model/riskmodel/structured.py`
3. `qlib/model/riskmodel/shrink.py`
4. `qlib/model/riskmodel/poet.py`
5. `qlib/contrib/strategy/optimizer/`
6. `examples/portfolio/config_enhanced_indexing.yaml`

### Questions

- What data is used to estimate covariance?
- How do structured, shrinkage, and POET estimators differ conceptually?
- How are forecast scores, benchmark weights, risk exposure, and constraints combined?
- Where does optimization end and order generation begin?

### Exercise

Use a tiny covariance matrix and prediction vector to describe an optimization problem by hand. Then map each term to the enhanced-indexing configuration.

---

# Part IX — Capstone

## Capstone: Build and explain a small research system

Build a workflow that is intentionally simple but complete.

### Requirements

1. Use a small market and time range.
2. Define fewer than ten understandable features.
3. Define one explicit forward-return label.
4. Use fitted preprocessing without future leakage.
5. Train a baseline and one stronger model.
6. Save predictions and model artifacts.
7. Evaluate IC and rank IC.
8. Use the same strategy and costs for both models.
9. Run a portfolio backtest.
10. Compare pre-cost and after-cost results.
11. Add at least one test for every custom component.
12. Provide both YAML-driven and Python-driven versions.

### Required explanation

Write `self_learn/capstone_report.md` answering:

- What information is available at each timestamp?
- Where could leakage occur, and how was it prevented?
- What exactly does the model output mean?
- How are scores converted into holdings and orders?
- Which exchange rules or costs reject or alter intended trades?
- Which metrics measure the model, strategy, and execution separately?
- Which artifacts are sufficient to reproduce and diagnose the run?
- Which result changed when only the model changed?
- Which result changed when only the strategy changed?

### Final oral/code review

Starting from the YAML file, point to the code responsible for:

1. Object construction
2. Feature loading
3. Expression evaluation
4. Processor fitting
5. Segment selection
6. Model fitting
7. Prediction generation
8. Artifact storage
9. Portfolio decisions
10. Order execution
11. Position updates
12. Metric computation

If you can do this and modify one component without unintentionally changing the others, you understand Qlib's architecture at a practical level.

---

# Suggested schedule

The schedule is deliberately flexible. “Session” means one focused study block, not necessarily one day.

| Phase | Modules | Suggested sessions | Outcome |
|---|---:|---:|---|
| Orientation | 0–1 | 2–3 | Package map and YAML-to-object understanding |
| Data infrastructure | 2–4 | 5–7 | Trace raw data through `D.features()` |
| Dataset preparation | 5–7 | 5–7 | Build leakage-safe model inputs |
| Modeling/workflow | 8–9 | 4–6 | Train a model and recover recorded artifacts |
| Trading/backtesting | 10–13 | 7–10 | Trace scores into executed trades and reports |
| Integration | 14–15 | 5–8 | Run and explain the full workflow two ways |
| Extension/testing | 16–17 | 5–8 | Build replaceable custom components |
| Advanced topics | 18–23 | As needed | Specialize after mastering the core |
| Capstone | — | 5–10 | Independent end-to-end system and explanation |

Do not advance based only on reading completion. Advance when you can pass the checkpoint and show a small working artifact.

# Progress tracker

## Core architecture

- [ ] Module 0: Package map
- [ ] Module 1: Initialization and object construction
- [ ] Module 2: Physical data and storage
- [ ] Module 3: Providers and `D`
- [ ] Module 4: Expressions, operators, and cache
- [ ] Module 5: Data loaders
- [ ] Module 6: Processors and handlers
- [ ] Module 7: Datasets, segments, and weights
- [ ] Module 8: Models
- [ ] Module 9: Experiments and record templates
- [ ] Module 10: Signals and strategies
- [ ] Module 11: Decisions and exchange
- [ ] Module 12: Executors and backtest loop
- [ ] Module 13: Evaluation and reports
- [ ] Module 14: Full YAML trace
- [ ] Module 15: Workflow by code
- [ ] Module 16: Custom components
- [ ] Module 17: Testing and debugging

## Advanced architecture

- [ ] Module 18: Rolling tasks
- [ ] Module 19: Online workflow
- [ ] Module 20: Nested/high-frequency execution
- [ ] Module 21: Reinforcement learning
- [ ] Module 22: Meta-learning
- [ ] Module 23: Risk and optimization
- [ ] Capstone completed

# Fast reference: where to look when confused

| Question | First place to inspect |
|---|---|
| Why is initialization failing? | `qlib/__init__.py`, `qlib/config.py` |
| How was this configured class created? | `qlib/utils/__init__.py`, `qlib/cli/run.py` |
| Where did this market-data value come from? | `qlib/data/data.py`, `qlib/data/storage/` |
| How was this feature calculated? | `qlib/data/base.py`, `qlib/data/ops.py` |
| Why does the model see these columns? | `qlib/data/dataset/loader.py`, handler configuration |
| Why did preprocessing change the value? | `qlib/data/dataset/processor.py`, `handler.py` |
| Why is this row in train or test? | Dataset `segments` and `DatasetH.prepare()` |
| How is model output produced? | `qlib/model/base.py`, concrete model `predict()` |
| Where are predictions saved? | `SignalRecord` in `qlib/workflow/record_temp.py` |
| Why was this stock selected? | Concrete strategy implementation |
| Why did an order not execute? | `exchange.py`, `executor.py`, account cash/position |
| How was this metric computed? | `qlib/contrib/evaluate*.py`, `backtest/report.py` |
| Why does YAML behave differently from my code? | Compare constructed objects and resolved kwargs |

# Guiding principle

When debugging or learning Qlib, always identify the current boundary:

```text
storage boundary
    -> retrieval boundary
    -> preprocessing boundary
    -> dataset boundary
    -> model boundary
    -> signal boundary
    -> decision boundary
    -> execution boundary
    -> reporting boundary
```

Most confusion comes from mixing responsibilities across two boundaries. Once the input and output at each boundary are made explicit, the package becomes much easier to understand.
