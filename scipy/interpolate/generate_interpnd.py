#!/usr/bin/env python
import subprocess
subprocess.call(['cython', '-I', '../..', '-o', 'interpnd.c', 'interpnd.pyx'])
