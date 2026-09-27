#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# mark1-runtime
# Copyright (C) 2026 SnapKitty Collective
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#


# MARK-I s390x build pipeline: runs inside proot+qemu user-mode emulation.
Z=~/workspace/mark1-runtime/zroot
P() { sudo proot -q /usr/bin/qemu-s390x-static -r $Z -b /dev -b /proc -w / "$@"; }
set -x
P /usr/bin/apt-get update
P /usr/bin/apt-get install -y gnat
echo "=== gnat version (s390x) ==="
P /usr/bin/gnat --version | head -2
echo "=== build ==="
P /bin/bash -c 'cd /root/mark1/src && /usr/bin/gnatmake -gnat05 -gnata -gnatW8 -O2 mark1_runtime 2>&1'
echo "=== file ==="
P /usr/bin/file /root/mark1/src/mark1_runtime
echo "=== run ==="
P /bin/bash -c 'cd /root/mark1/src && ./mark1_runtime'
echo "=== PIPELINE DONE ==="
