# Copyright 2015-2024 David Hadka
#
# This file is part of Rhodium, a Python module for robust decision
# making and exploratory modeling.
#
# Rhodium is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Rhodium is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Rhodium.  If not, see <http://www.gnu.org/licenses/>.
import os
import sys

def promptToRun(message, default="yes"):
    if os.getenv("RHODIUM_NO_PROMPT"):
        response = ""
    else:
        if default == "yes":
            prompt = "[Y/n]"
        elif default == "no":
            prompt = "[y/N]"
        else:
            raise ValueError("invalid default answer")

        print(message + " " + prompt + " ", end='')
        sys.stdout.flush()

        response = sys.stdin.readline().strip()

    if response == "":
        response = default[0]

    if response == "y" or response == "Y":
        return True
    else:
        return False
