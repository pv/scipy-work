#!/usr/bin/env python
"""
boost_prune_special.py BOOST_INC_DIR

Remove Boost header files not required by Boost special function library.

"""
import os
import re
import subprocess
import argparse

def main():
    p = argparse.ArgumentParser(usage=__doc__.strip())
    p.add_argument('boost_inc_dir', metavar='BOOST_INC', type=str,
                   help="boost include directory")
    p.add_argument('--remove', dest='remove', action='store_true',
                   default=False,
                   help="Remove the unnecessary files")
    args = p.parse_args()

    process(args.boost_inc_dir, remove=args.remove)

def process(boost_inc_dir, remove=False):
    boost_inc_dir = os.path.normpath(os.path.abspath(boost_inc_dir))
    inc_path = os.path.join(boost_inc_dir, os.pardir)

    required_headers = set()

    hdr_re = re.compile(r'^# \d+ "(.*?)".*$')

    cmd = ['g++', '-E', '-I' + inc_path,
           os.path.join(boost_inc_dir, 'math/special_functions.hpp')]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    for line in p.stdout:
        m = hdr_re.match(line)
        if m:
            hdr = m.group(1)
            hdr = os.path.normpath(os.path.abspath(hdr))
            if hdr.startswith(boost_inc_dir + os.sep):
                required_headers.add(hdr)
    p.terminate()

    preserve_dirs = ['config', 'compatibility', 'type_traits',
                     'smart_ptr', 'detail']

    for root, dirs, files in os.walk(boost_inc_dir, topdown=False):
        for fn in files:
            fn = os.path.normpath(os.path.join(root, fn))

            skip = any(('boost' + os.sep + d + os.sep) in fn
                       for d in preserve_dirs)
            if skip:
                continue

            if fn not in required_headers:
                print fn
                if remove:
                    os.unlink(fn)

        for dn in dirs:
            dn = os.path.join(root, dn)
            if not os.listdir(dn):
                print dn
                if remove:
                    os.rmdir(dn)

if __name__ == "__main__":
    main()
