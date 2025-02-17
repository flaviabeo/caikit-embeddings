# Copyright The Caikit Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Standard
from typing import List

# First Party
from caikit.core import TaskBase, task

# Local
from ...data_model import ListOfVector1D, Vector1D


@task(
    required_parameters={"text": str},
    output_type=Vector1D,
)
class EmbeddingTask(TaskBase):
    """Return a text embedding for the input text string"""


@task(
    required_parameters={"texts": List[str]},
    output_type=ListOfVector1D,
)
class EmbeddingTasks(TaskBase):
    """Return a text embedding for each text string in the input list"""
