#!/bin/bash


. ./env.bash

init_paths

ivf_build_prefix=${default_ivf_build_prefix}

build_ivf_hack ${ivf_build_prefix} lums/tmp/part

