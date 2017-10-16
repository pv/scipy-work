"""run-valgrind.py [OPTIONS] [PYTEST_ARGS]

This is a tool to maintain and run scipy with valgrind against a set
of suppression files.

A suppression file is a list of known memory errors, many of which can
be false positives.

We scan valgrind log to identify potential erros related to scipy, and
maintain a suppression db for non-scipy errors.

Examples:

# Find scipy related errors and update the non-scipy suppression db.

python tools/run-valgrind.py --python=python3-debug/bin/python3-debug scipy/_lib/tests/test__gcutils.py

# Find scipy related errors and replace the non-scipy suppression db.

python tools/run-valgrind.py --update-supp=replace --python=python3-debug/bin/python3-debug scipy/_lib/tests

# Find scipy related errors and do not update the non-scipy suppression db.

python tools/run-valgrind.py --update-supp=no --python=python3-debug/bin/python3-debug scipy/_lib/tests

The errors and suppression files for scipy and non-scipy entries are
stored in the valgrind/ directory for the test case.

Selected rules can be manually merged into the default file
valgrind-suppression in valgrind directory.

"""
from __future__ import print_function

import os
import sys
import time

from argparse import ArgumentParser
from subprocess import call
from tempfile import mkstemp


TOOL_DIR = os.path.abspath(os.path.dirname(__file__))
DEFAULT_PYTEST_ARGS = ["--pyarg", "scipy"]


def main():
    ap = ArgumentParser(usage=__doc__.strip())
    ap.add_argument("--update-supp", choices=["merge", "replace", "no"],
                    default='no',
                    help='strategy for merging non-scipy valgrind errors')
    ap.add_argument("--prefix", default=os.path.join(TOOL_DIR, '..', 'valgrind'))
    ap.add_argument("--valgrind", default='valgrind')
    ap.add_argument("--suppressions", default=[], action='append')
    ap.add_argument("pytest_args", metavar="PYTEST_ARGS", nargs="*",
                    help="pytest arguments")
    args = ap.parse_args()

    rules(prefix=args.prefix,
          valgrind=args.valgrind,
          suppressions=args.suppressions,
          update_suppressions=args.update_supp,
          pytest_args=args.pytest_args)

    return 0


def run_valgrind(valgrind, opts, payload, suppressions, logfilename):
    valgrind_opts = opts + [
        '--log-file=%s' % logfilename,
    ]

    for fn in suppressions:
        if not os.path.exists(fn):
            raise ValueError("file %s not found" % fn)

    suppressions = [ '--suppressions=%s' % fn for fn in suppressions ]
    cmdline = [valgrind] + valgrind_opts + suppressions + payload

    env = dict(os.environ)
    env['PYTHONMALLOC'] = 'malloc'
    call(cmdline, env=env)

    with open(logfilename, 'r') as ff:
        r = ff.read()

    if 'FATAL' in r:
        print(r)
        raise RuntimeError("Valgrind failed with")

    return r


def find_suppression_files(prefix):
    path_suppr = os.path.join(prefix, 'valgrind-suppression')
    if os.path.exists(path_suppr):
        return [path_suppr]
    else:
        return []


def find_local_suppression_file(prefix):
    if not os.path.isdir(prefix):
        return None
    path_suppr = os.path.join(prefix, 'valgrind-suppression')
    return path_suppr


def rules(prefix, valgrind, suppressions, update_suppressions,
          pytest_args):
    if sys.version_info[:2] < (3, 6):
        print("WARNING: Python < 3.6 do not support PYTHONMALLOC env variable.\n"
              "         This will result to lots of garbage valgrind hits.")

    test_command = [sys.executable, '-mpytest']
    if pytest_args:
        test_command += list(pytest_args)
    else:
        test_command += DEFAULT_PYTEST_ARGS
    print(test_command)

    opts = [
        '--show-leak-kinds=all',
        '--leak-check=full',
        '--num-callers=40',
        '--error-limit=no',
        '--fullpath-after=', # for scipy detection
        '--gen-suppressions=all', # for suppressions generation
    ]

    t0 = time.time()

    if not os.path.isdir(prefix):
        os.makedirs(prefix)

    suppressions = list(suppressions)
    all_test_suppr = [os.path.join(TOOL_DIR, 'valgrind-scipy.supp'),
                      os.path.join(TOOL_DIR, 'valgrind-python.supp')] + find_suppression_files(prefix)
    print("all suppression files", all_test_suppr)

    suppressions = suppressions + all_test_suppr

    print("using suppression files", suppressions)

    if update_suppressions == "replace":
        local_supp = find_local_suppression_file(prefix)
        while local_supp in suppressions:
            suppressions.remove(local_supp)

    print("running valgrind with the tests")

    logfilename = os.path.join(prefix, 'valgrind.log')
    log = run_valgrind(valgrind, opts, test_command, suppressions, logfilename)
    vlog = ValgrindLog.from_string(log)

    scipy_errors = ValgrindLog()
    non_scipy_errors = ValgrindLog()

    for section in vlog:
        if section.is_heap_summary():
            print('\n'.join(section))
        if section.is_error_summary():
            print('\n'.join(section))
        if section.is_leak_summary():
            print('\n'.join(section))

        sc = section.get_scipy_related()
        if sc is not None:
            scipy_errors.append(section)
        else:
            non_scipy_errors.append(section)

    print('Found %d valgrind anomalies that appeared to be related to scipy' % len(scipy_errors))
    if len(scipy_errors):
        print(str(scipy_errors))

    print('Found %d valgrind anomalies that appeared to be unrelated to scipy' % len(non_scipy_errors))

    with open(os.path.join(prefix, 'scipy.log'), 'w') as ff:
        ff.write(str(scipy_errors))
    print("Scipy error log ", os.path.join(prefix, 'scipy.log'))

    with open(os.path.join(prefix, 'scipy.supp'), 'w') as ff:
        ff.write(str(scipy_errors.get_suppression_db()))
    print("Scipy suppression rules", os.path.join(prefix, 'scipy.supp'))

    with open(os.path.join(prefix, 'nonscipy.log'), 'w') as ff:
        ff.write(str(non_scipy_errors))
    print("Non-scipy error log ", os.path.join(prefix, 'nonscipy.log'))

    with open(os.path.join(prefix, 'nonscipy.supp'), 'w') as ff:
        ff.write(str(non_scipy_errors.get_suppression_db()))
    print("Non-scipy suppression rules ", os.path.join(prefix, 'nonscipy.supp'))

    if update_suppressions != 'no':
        local_supp = find_local_suppression_file(prefix)
        newdb = non_scipy_errors.get_suppression_db()

        print("Found %d suppression rules" % len(newdb))

        if update_suppressions == 'replace':
            pass
        elif update_suppressions == 'merge':
            try:
                db = SuppressionDB.fromfile(local_supp)
            except IOError:
                db = SuppressionDB()

            print("Merging existing %d suppression into %d rules" %( len(db), len(newdb)))

            newdb.update(db)

        print("Written %d suppression rules to %s" % (len(newdb), local_supp))
        with open(local_supp, 'w') as ff:
            ff.write(str(newdb))

    t1 = time.time()
    print("Testing used %g seconds" % (t1 - t0,))


class ValgrindSection(list):
    def __init__(self):
        self.supp_rule = []

    @classmethod
    def from_list(cls, list, supp_rule):
        self = ValgrindSection()
        self.extend(list)
        self.supp_rule = supp_rule
        return self

    def is_warn(self):
        if len(self) == 0: return False
        return self[0].startswith('Warning:')

    def is_heap_summary(self):
        if len(self) == 0: return False
        for line in self:
            if line.startswith("HEAP SUMMARY:"):
                return True

    def is_leak_summary(self):
        if len(self) == 0: return False
        for line in self:
            if line.startswith("LEAK SUMMARY:"):
                return True

    def is_error_summary(self):
        if len(self) == 0: return False
        for line in self:
            if line.startswith("ERROR SUMMARY:"):
                return True

    def is_entry(self):
        return len(self.supp_rule) > 0

    def __str__(self):
        return '\n'.join(self)

    def get_scipy_related(self):
        if not self.is_entry(): return None
        r = []
        for line in self:
            if 'scipy/' in line:
                r.append(line)
        if len(r) is 0: return None
        return ValgrindSection.from_list(r, self.supp_rule)

    def format_suppression(self):
        return '\n'.join(self.supp_rule)


class ValgrindLog(list):
    def __init__(sections):
        pass

    @classmethod
    def from_string(kls, log):
        sections = kls()
        section_start = True
        section = ValgrindSection()
        for i, line in enumerate(log.split('\n')):
            if line.startswith('=='):
                pid, line = line.split(' ', 1)
            else:
                pid = None

            if pid is not None:
                if section_start:
                    sections.append(section)
                    section = ValgrindSection()
                    section_start = False

                if len(line.strip()) == 0:
                    section_start = True
                else:
                    section.append(line.strip())

            if pid is None:
                section.supp_rule.append(line.rstrip())
        return sections

    def get_suppression_db(self):
        db = SuppressionDB()
        for section in self:
            db.add(section.format_suppression())
        return db

    @classmethod
    def fromfile(cls, filename):
        return cls(open(filename).read())

    def __str__(self):
        return '\n\n'.join([str(i) for i in self])


class SuppressionDB(set):
    @classmethod
    def fromfile(cls, filename):
        self = cls()
        rule = None
        with open(filename) as ff:
            for i , line in enumerate(ff.readlines()):
                if len(line.strip()) == 0:
                    continue
                if line.strip().startswith('{'):
                    rule = []

                rule.append(line.rstrip())

                if line.strip().startswith('}'):
                    self.add('\n'.join(rule))

        return self

    def __str__(self):
        return '\n\n'.join(sorted(self))


if __name__ == "__main__":
    sys.exit(main())
