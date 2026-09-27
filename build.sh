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


# MARK-I Runtime build. Ada 2007 = ISO/IEC 8652:2007 (the Ada 2005 revision).
set -e
cd "$(dirname "$0")/src"
gnatmake -gnat05 -gnata -gnatW8 -O2 mark1_runtime.adb -o ../mark1_runtime
echo "--- running ---"
../mark1_runtime
