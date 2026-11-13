# ZN Order Flow

A model of the orders arriving in 10-year US Treasury note futures (ZN) on CME: when they arrive, how big they are and which way they go. Fitted on six months of trades, validated on four and tested on two. The findings are in `report.md`.

## Setup

- Python 3.12
- Install the packages

  ```
  pip install numpy pandas scipy numba duckdb matplotlib pyarrow databento
  ```

- Get your own Databento API key. The data is paid and is not in this repo
  - Sign up at [databento.com](https://databento.com) and copy the key from the portal
  - Set it in your shell

    ```
    export DATABENTO_API_KEY=your-key
    ```

- Download a year of ZN trades with quotes. Databento charges for this, so check the price in the portal first. Months already on disk are skipped

  ```
  python analysis/fetch.py GLBX.MDP3 ZN.v.0 2025-11-01 2026-10-31 data/databento/ZN_tbbo
  ```

## Run

- Each stage reads the raw files and writes its tables and figures to `output/`

  ```
  python analysis/run.py fit
  python analysis/run.py quoting
  python analysis/run.py validate
  python analysis/run.py test
  ```

- `fit` fits every model on the fit months and saves it
- `quoting` adds the quoting thresholds to the saved model
- `validate` and `test` score the saved model on unseen months
- To change the data or the split, edit `analysis/config.json`

## Not Included

- The Databento data. Its licence does not allow sharing
- The course's assignment guide, lecture slides and data package. They belong to UNSW
- The data behind the first version on GLD, so `exploration/` is there to read and will not run
