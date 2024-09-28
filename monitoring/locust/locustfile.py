"""
Run load tests: locust -f monitoring/locust/locustfile.py
"""
import math
import json
import random
import logging
import os
from locust import (
    HttpUser,
    LoadTestShape,
    task,
    run_single_user,
    events,
    between,
    SequentialTaskSet
)


@events.request.add_listener
def request_handler(
    request_type: str,
    name: str,
    response_time: int, response_length: int,
    response: object,
    context: dict,
    exception: Exception,
    start_time: float,
    url: str,
):
    if exception:
        print(f"Request of type {request_type} to {name} \
            failed with exception {exception}")
    else:
        print(f"Successfully made a request to: {name}")
        print(f"The response was {response.text} in {response_time} ms")

class ConvUser(HttpUser):
    abstract = True
    host = "http://api.3dconvad.trendatre3.duckdns.org/"

    def sanity_check(self):
        try:
            with self.client.get('', catch_response=True) as resp:
                if resp.status_code != 200:
                    raise Exception(resp.status_code)
                else:
                    resp.success()
        except Exception as e:
            print(f"Sanity check failed with status code {resp.status_code} throwing exception: {e}")
    @task
    class PredictionTask(SequentialTaskSet):

        @task
        def post_prediction(self):
            subject = random.choice(os.listdir("data/samples"))
            payload = {}
            files_dir = os.listdir(os.path.join("data/samples", subject))

            if random.random() > 0.5:
                brainscan = list(filter(lambda x: x.endswith(".nii.gz"), files_dir))[0]
                brainmask = list(filter(lambda x: x.endswith("_brainmask.mgz"), files_dir))[0]
                files = [
                ('brainscan',(brainscan, open(os.path.join("data/samples", subject, brainscan), 'rb'), 'application/gzip')),
                ('brainmask',(brainmask, open(os.path.join("data/samples", subject, brainmask), 'rb'), 'application/octet-stream'))
                ]
            else:
                brainscan = list(filter(lambda x: x.endswith("_brain.mgz"), files_dir))[0]
                files = [
                ('brainscan',(brainscan, open(os.path.join("data/samples", subject, brainscan), 'rb'), 'application/octet-stream')),
                ]
            with self.client.post("predict", data=payload, files=files) as response:
                self.task_id = json.loads(response.text)["task_id"]

        @task
        def get_status(self):
            while True:
                with self.client.get(f"predict/{self.task_id}") as resp:
                    self.status_code = resp.status_code
                    self.wait()
                    if self.status_code != 202:
                        break

        @task
        def get_result(self):
            if self.status_code != 404:
                self.client.get(f"results/{self.task_id}")

        @task
        def stop(self):
            self.interrupt()

    def on_start(self):
        self.sanity_check()


class StandardUser(ConvUser):
    wait_time = between(5, 10)


class CompulsiveUser(ConvUser):
    wait_time = between(1, 2)


#run_single_user(user_class=CompulsiveUser, loglevel="INFO", )

class StepLoadShape(LoadTestShape):
    """
    A step load shape
    Keyword arguments:
        step_time -- Time between steps
        step_load -- User increase amount at each step
        spawn_rate -- Users to stop/start per second at every step
        time_limit -- Time limit in seconds
    """

    step_time = 30
    step_load = 2
    spawn_rate = 1
    time_limit = 600

    def tick(self):
        run_time = self.get_run_time()

        if run_time > self.time_limit:
            return None

        current_step = math.floor(run_time / self.step_time) + 1
        return (current_step * self.step_load, self.spawn_rate)

# class DoubleWave(LoadTestShape):
#     """
#     A shape to imitate some specific user behaviour. In this example, midday
#     and evening meal times. First peak of users appear at time_limit/3 and
#     second peak appears at 2*time_limit/3
#     Settings:
#         min_users -- minimum users
#         peak_one_users -- users in first peak
#         peak_two_users -- users in second peak
#         time_limit -- total length of test
#     """

#     min_users = 2
#     peak_one_users = 5
#     peak_two_users = 10
#     time_limit = 600

#     def tick(self):
#         run_time = round(self.get_run_time())

#         if run_time < self.time_limit:
#             user_count = (
#                 (self.peak_one_users - self.min_users)
#                 * math.e ** -(((run_time / (self.time_limit / 10 * 2 / 3)) - 5) ** 2)
#                 + (self.peak_two_users - self.min_users)
#                 * math.e ** -(((run_time / (self.time_limit / 10 * 2 / 3)) - 10) ** 2)
#                 + self.min_users
#             )
#             return (round(user_count), round(user_count))
#         else:
#             return None

