"""Doorbell battery API. Start from the repository root with uvicorn app.main:app."""

from time import perf_counter_ns

import numpy as np
import torch
from fastapi import APIRouter, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from pydantic import BaseModel, ConfigDict, Field

from app.artifacts import load_model
from app.model import FEATURES, describe
from app.sensors import BOUNDS

cfg, net, MU, SD = load_model()
torch.set_num_threads(1)


class Reading(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    battery_v: float = Field(ge=BOUNDS["battery_v"][0], le=BOUNDS["battery_v"][1])
    temp_c: float = Field(ge=BOUNDS["temp_c"][0], le=BOUNDS["temp_c"][1])
    rings_per_day: float = Field(ge=BOUNDS["rings_per_day"][0], le=BOUNDS["rings_per_day"][1])
    wifi_rssi_dbm: float = Field(ge=BOUNDS["wifi_rssi_dbm"][0], lt=BOUNDS["wifi_rssi_dbm"][1])
    age_months: float = Field(ge=BOUNDS["age_months"][0], le=BOUNDS["age_months"][1])


class Batch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rows: list[Reading] = Field(min_length=1, max_length=10000)


class TimedRoute(APIRoute):
    """Start the clock before FastAPI reads JSON and validates the request body."""

    def get_route_handler(self):
        handler = super().get_route_handler()

        async def timed(request):
            request.state.started = perf_counter_ns()
            return await handler(request)

        return timed


app = FastAPI(title="Doorbell battery API")
predictions = APIRouter(route_class=TimedRoute)


@app.exception_handler(RequestValidationError)
async def bad_reading(request, exc):
    # Echo only structured reasons. Invalid input can itself contain unencodable NaN.
    detail = [{key: error[key] for key in ("loc", "msg", "type")} for error in exc.errors()]
    return JSONResponse(status_code=422, content={"error": "bad reading", "detail": detail})


@app.get("/health")
def health():
    return {
        "status": "ok",
        "team": cfg["team"],
        "params": sum(p.numel() for p in net.parameters()),
        "val_mae": cfg["val_mae"],
    }


@app.get("/model_info")
def model_info():
    return describe(net)


def run(data, started):
    parsed = perf_counter_ns()
    bulk = isinstance(data, Batch)
    rows = data.rows if bulk else [data]
    x = np.array([[getattr(row, key) for key in FEATURES] for row in rows], dtype=np.float32)
    x = torch.from_numpy((x - MU) / SD)
    normalized = perf_counter_ns()
    # Fixed-size blocks give single and batch rows the same float32 reduction path,
    # including at the half-centiday rounding boundary.
    block = 32
    with torch.inference_mode():
        x = torch.nn.functional.pad(x, (0, 0, 0, -len(rows) % block))
        values = (
            torch.cat([net(part) for part in x.split(block)])[: len(rows)].clamp_min(0).tolist()
        )
    inferred = perf_counter_ns()
    days = [round(v, 2) for v in values]
    response = {"days_left": days if bulk else days[0]}
    built = perf_counter_ns()
    response["timings_ms"] = {
        "parse": (parsed - started) / 1e6,
        "normalise": (normalized - parsed) / 1e6,
        "inference": (inferred - normalized) / 1e6,
        "build": (built - inferred) / 1e6,
        "server_total": (built - started) / 1e6,
    }
    return response


@predictions.post("/predict")
async def predict(reading: Reading, request: Request):
    return run(reading, request.state.started)


@predictions.post("/predict_batch")
async def predict_batch(batch: Batch, request: Request):
    return run(batch, request.state.started)


app.include_router(predictions)
