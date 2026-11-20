# Team 02 acceptance checklist

| Instructor check | Implementation | Independent check |
|---|---|---|
| Health | app/main.py loads the packaged checkpoint and reports team, validation MAE and parameter count | tests/test_service.py checks a fresh server and its response |
| Accuracy | train.py separates the final holdout before candidate selection, fits transforms on training rows and exports weights | tests/test_service.py reconstructs the holdout from the team CSV and scores single HTTP predictions |
| Layer table | app/model.py walks the Linear modules | tests/test_service.py calculates every field from checkpoint tensor shapes and checks SRAM limits |
| Broken readings | app/main.py validates strict finite JSON numbers against shared limits | tests/test_service.py sends malformed bodies, sensor glitches and invalid batches |
| Stage timing | app/main.py measures consecutive intervals with a monotonic counter | tests/test_service.py checks exact keys, finite values and their sum |
| Batch | app/main.py applies the same feature order and prediction pipeline to one or many readings | tests/test_service.py compares ordered responses for 1, 50 and 400 rows |
| Deployment | model/config.json and model/model.pt load without training | tests/test_service.py launches a separate Uvicorn process and verifies repeatability after restart |

## Starter audit

The sole configured CSV is team_02_train.csv. The roster named in the README is absent. The starter split was deterministic, but its glitch mask retained every row, input statistics were fixed at zero and one, and the linear baseline was missing. No weights were supplied. The network itself was complete. The API accepted unconstrained/coercible values, returned an empty model table and omitted three timing stages. Its raw Request endpoints also left the request bodies out of the interactive docs. The original benchmark and devcontainer are retained.
