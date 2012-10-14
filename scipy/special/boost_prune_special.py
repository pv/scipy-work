#!/usr/bin/env python
"""
boost_prune_special.py [BOOST_DIR]

Remove Boost header files not required by Boost special function library.

"""
import os
import re
import subprocess
import argparse

def main():
    p = argparse.ArgumentParser(usage=__doc__.strip())
    p.add_argument('boost_dir', metavar='BOOST_DIR', type=str,
                   default='boost', nargs='?',
                   help="boost base directory")
    p.add_argument('--remove', dest='remove', action='store_true',
                   default=False,
                   help="Remove the unnecessary files")
    args = p.parse_args()

    process(args.boost_dir, remove=args.remove)

def process(boost_dir, remove=False):
    boost_dir = os.path.normpath(os.path.abspath(boost_dir))
    boost_inc_dir = os.path.join(boost_dir, 'boost')

    required_headers = set()

    cmd = ['bcp', '--list', '--boost=' + boost_dir,
           'boost/math/special_functions.hpp',
           'boost/math/distributions.hpp',
           ]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    for line in p.stdout:
        line = line.strip()
        if line.startswith('*'):
            print line
            continue
        if line:
            hdr = os.path.join(boost_dir, line)
            hdr = os.path.normpath(os.path.abspath(hdr))
            required_headers.add(hdr)
    p.communicate()

    if p.returncode != 0:
        raise RuntimeError("bcp did not exit successfully")

    for root, dirs, files in os.walk(boost_inc_dir, topdown=False):
        for fn in files:
            fn = os.path.normpath(os.path.join(root, fn))

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
