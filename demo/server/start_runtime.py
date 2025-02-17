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
from os import path
import sys

# First Party
import alog

# Add the runtime/library to the path
sys.path.append(
    path.abspath(path.join(path.dirname(__file__), "../../"))
)

# Local
import caikit_embeddings
import caikit
from caikit.runtime.__main__ import main

# Load configuration for model(s) serving
CONFIG_PATH = path.realpath(
    path.join(path.dirname(__file__), "config.yml")
)
caikit.configure(CONFIG_PATH)

alog.configure(default_level="debug")

# Load the model(s) in gRPC and REST servers
main()
