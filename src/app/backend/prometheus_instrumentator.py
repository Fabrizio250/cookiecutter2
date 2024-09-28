"""
Module to instantiate custom instrumentator
"""
import json
import os
from typing import Callable

import user_agents
from prometheus_client import Counter, Summary
from prometheus_fastapi_instrumentator import Instrumentator, metrics
from prometheus_fastapi_instrumentator.metrics import Info

NAMESPACE = os.environ.get("METRICS_NAMESPACE", "backend")
SUBSYSTEM = os.environ.get("METRICS_SUBSYSTEM", "model")

instrumentator = Instrumentator(
    should_group_status_codes=True,
    should_ignore_untemplated=False,
    should_respect_env_var=False,
    should_instrument_requests_inprogress=True,
    excluded_handlers=["/metrics"],
    inprogress_name="fastapi_inprogress",
    inprogress_labels=True,
)

# ==============================
# ====== STANDARD METRICS ======
# ==============================

instrumentator.add(
    metrics.request_size(
        should_include_handler=True,
        should_include_method=True,
        should_include_status=True,
        metric_namespace=NAMESPACE,
        metric_subsystem=SUBSYSTEM,
    )
)
instrumentator.add(
    metrics.response_size(
        should_include_handler=True,
        should_include_method=True,
        should_include_status=True,
        metric_namespace=NAMESPACE,
        metric_subsystem=SUBSYSTEM,
    )
)
instrumentator.add(
    metrics.latency(
        should_include_handler=True,
        should_include_method=True,
        should_include_status=True,
        metric_namespace=NAMESPACE,
        metric_subsystem=SUBSYSTEM,
    )
)
instrumentator.add(
    metrics.requests(
        should_include_handler=True,
        should_include_method=True,
        should_include_status=True,
        metric_namespace=NAMESPACE,
        metric_subsystem=SUBSYSTEM,
    )
)


# ============================
# ====== CUSTOM METRICS ======
# ============================
def requests_user_agent(
        metric_name: str = "user_agent_total",
        metric_doc: str = "User agent for the requests",
        metric_namespace: str = "",
        metric_subsystem: str = ""
) -> Callable[[Info], None]:
    """
    Keep track of the User-Agents from which requests are done
    """
    METRIC = Counter(
        metric_name,
        metric_doc,
        labelnames=("browser", 'version'),
        namespace=metric_namespace,
        subsystem=metric_subsystem,
    )

    def instrumentation(info: Info) -> None:
        if info.modified_handler == "/predict":
            try:
                user_agent = user_agents.parse(info.request.headers.get("User-Agent"))
                METRIC.labels(user_agent.browser.family, user_agent.browser.version_string).inc()
            except KeyError:
                pass

    return instrumentation


def prediction_class(
        metric_name: str = "predicted_classes",
        metric_doc: str = "Class predicted (negative or positive)",
        metric_namespace: str = "",
        metric_subsystem: str = ""
) -> Callable[[Info], None]:
    """
    Keep track of the predicted classes
    """
    METRIC = Counter(
        metric_name,
        metric_doc,
        labelnames=("class",),
        namespace=metric_namespace,
        subsystem=metric_subsystem,
    )

    def instrumentation(info: Info) -> None:
        if info.modified_handler == "/results/{task_id}":
            if info.response:
                try:
                    predicted_class = json.loads(info.response.body)["prediction"]['predicted_class']
                    predicted_class = 'positive' if int(predicted_class) == 1 else 'negative'
                    METRIC.labels(predicted_class).inc()
                except KeyError:
                    pass
    return instrumentation


def processing_time(
        metric_name: str = "processing_seconds",
        metric_doc: str = "processing time in seconds",
        metric_namespace: str = "",
        metric_subsystem: str = ""
) -> Callable[[Info], None]:
    """
    Keep track of the time required for processing and inference
    """
    METRIC = Summary(
        metric_name,
        metric_doc,
        labelnames=("time",),
        namespace=metric_namespace,
        subsystem=metric_subsystem,
    )

    def instrumentation(info: Info) -> None:
        if info.modified_handler == "/results/{task_id}":
            if info.response:
                response = json.loads(info.response.body)
                if response:
                    METRIC.labels('processing').observe(response.get("processing_seconds", 0))
                    METRIC.labels("inference").observe(response.get("inference_seconds", 0))

    return instrumentation


def attribution_iterations(
        metric_name: str = "attribution_iterations",
        metric_doc: str = "attribution iteration requested",
        metric_namespace: str = "",
        metric_subsystem: str = ""
) -> Callable[[Info], None]:
    """
    Keep track of the time required for processing and inference
    """
    METRIC = Summary(
        metric_name,
        metric_doc,
        namespace=metric_namespace,
        subsystem=metric_subsystem,
    )

    def instrumentation(info: Info) -> None:
        if info.modified_handler == "/predict":
            if info.response:
                response = json.loads(info.response.body)
                if response:
                    METRIC.observe(response.get("requested_iter", 1))

    return instrumentation


def payload_bytes(
        metric_name: str = "payload_bytes",
        metric_doc: str = "attribution iteration requested",
        metric_namespace: str = "",
        metric_subsystem: str = ""
) -> Callable[[Info], None]:
    """
    Keep track of the time required for processing and inference
    """
    METRIC = Summary(
        metric_name,
        metric_doc,
        namespace=metric_namespace,
        subsystem=metric_subsystem,
    )

    def instrumentation(info: Info) -> None:
        if info.modified_handler == "/predict":
            if info.request:
                METRIC.observe(float(info.request.headers.get("content-length", 0)))

    return instrumentation


instrumentator.add(payload_bytes(metric_namespace=NAMESPACE, metric_subsystem=SUBSYSTEM))
instrumentator.add(attribution_iterations(metric_namespace=NAMESPACE, metric_subsystem=SUBSYSTEM))
instrumentator.add(processing_time(metric_namespace=NAMESPACE, metric_subsystem=SUBSYSTEM))
instrumentator.add(prediction_class(metric_namespace=NAMESPACE, metric_subsystem=SUBSYSTEM))
instrumentator.add(requests_user_agent(metric_namespace=NAMESPACE, metric_subsystem=SUBSYSTEM))
