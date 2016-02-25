import fnmatch
import logging
import platform
import subprocess
from general.should_be_builtins import bad_value
import os
import jpype as jp

__author__ = 'peter'

_JAVA_CLASS_PATH = []


class JPypeConnection(object):
    """
    A JPype Connection, to be used in a "with" statement.

    Usage:

    with JPypeConnection():
        subprocess.call(['javac', './my_package/MyJavaClass.java'])
        c = jp.JClass('my_package.MyJavaClass')
        stuff = c.do_something()
    """

    def __init__(self, java_loc=None, root_dir = None):
        """
        :param root_dir: The root of your java code (if None, it is the current directory) (see os.getcwd)
        :param java_loc: The location of Java on the local machine.
        :return: A JPypeConnection object that takes care of shutting itself down on exit.
        """
        if root_dir is None:
            root_dir, _ = os.path.split(os.path.join(os.path.abspath(__file__)))
        if root_dir is None:
            root_dir = os.getcwd()
        self._java_loc = _guess_java_loc() if java_loc is None else java_loc
        self._root_dir = root_dir

    def __enter__(self):
        self._old_cd = os.getcwd()
        os.chdir(self._root_dir)
        if not jp.isJVMStarted():
            args = ['-ea']
            if len(_JAVA_CLASS_PATH)>0:
                args.append("-Djava.class.path=%s" % (':'.join(_JAVA_CLASS_PATH), ))
            jp.startJVM(self._java_loc, *args)
        global _JPYPE_SINGLETON
        _JPYPE_SINGLETON = jp
        return jp

    def __exit__(self, *args):
        """
        Here we try to intercept confusing errors and give hints as to what may be causing
        them.  Be a good person and add your hint here if you solve one.
        """
        if isinstance(args[1], jp.JavaException):
            ex = args[1]
            # For some reason the Java-stacktrace appears bottom-down, (while they're normally printed top-down in python)
            respectable_stacktrace = '\n'.join(ex.stacktrace().split('\n')[::-1])
            logging.error('%s\n%s:%s\n%s' % ('-'*20, ex.__class__, respectable_stacktrace, '-'*31))

            if args[0] is not None and args[0].__name__ == 'java.lang.RuntimeExceptionPyRaisable':
                    print '''
Error when calling the Java Code.  Did you:
- Put the package at the top of your java file?  ("package mypackagename;")
- Give the package the same name as the directory it's in?  (above example should be in ./mypackagename)
- (If constructing a class) does a class with this name really exist?  Are you referencing it by its
  package (e.g. "klass = jp.JClass('mypackagename.MyClassName')")
- Name the class the same as the file? (not sure if necessary)
'''
        elif len(args)==3 and isinstance(args[1], RuntimeError) and args[1].message.startswith('No matching overloads found.'):
            print "HEY YOU!  Did you remember to declare your constructor public?"
        # jp.shutdownJVM()  # Apperently, this doesn't need to be called.
        # And it prevents us from starting it again...
        os.chdir(self._old_cd)


def with_jpype(f):
    """
    Use this to decorate a function so that jpype is always started before it is called, and so that errors in java are
    caught and reported.
    :param f: The function to decorate.
    :return: A wrapper function that takes care of starting the JVM and connecting
    """

    def jpype_wrapper(*args, **kwargs):
        with JPypeConnection():
            out = f(*args, **kwargs)
        return out

    return jpype_wrapper


def register_java_class_path(class_path):
    """
    Add a new classpath to the set of class-paths for the JVM.  You need to do this before the
    JVM is started (which happends when you use JPypeConnection (see below), so generally, you
    want to call this function at import-time.

    path can be:
        A list, in which case each element is taken to be a path
        A string, consisting either of a single path or a set of paths
            separated by ":"
    """

    assert not jp.isJVMStarted(), "Too late!  JVM is already started.  You need to register class paths before starting the JVM (do it on import)."

    if isinstance(class_path, str):
        class_path = class_path.split(':')
    else:
        assert isinstance(class_path, (list, tuple)), "class_path should either be a string or a list/tuple"

    for cp in class_path:
        assert os.path.exists(cp), "Java Class path: %s does not exist" % (cp, )

    global _JAVA_CLASS_PATH
    _JAVA_CLASS_PATH += class_path


def find_jars(path):
    # Thanks kutschkem from https://github.com/originell/jpype/issues/150
    for root, _, files in os.walk(path):
        for items in fnmatch.filter(files, "*.jar"):
            yield os.path.join(root, items)


def _guess_java_loc():
    """
    Try to figure out the java loc on the machine.
    :return:
    """
    system = platform.system()
    if system == 'Darwin':
        # guess = '/Library/Java/JavaVirtualMachines/jdk1.7.0_79.jdk/Contents/MacOS/libjli.dylib'
        guess = '/Library/Java/JavaVirtualMachines/jdk1.8.0_66.jdk/Contents/MacOS/libjli.dylib'
    else:
        raise NotImplementedError('Only know how to do this on mac right now.')
    assert os.path.exists(guess), "Hmmm.  It's not here.  Install Java or update this function to be more clever at finding it."
    return guess


_compiled_srcs = set()


def jpype_compile(source_files, force_recompile = False):

    if isinstance(source_files, str):
        source_files = source_files.split(':')
    if isinstance(source_files, list):
        source_files = tuple(source_files)

    if source_files in _compiled_srcs and not force_recompile:
        return

    for src in source_files:
        assert os.path.exists(src), 'File "%s" does not exist!' % (src, )
        if src.endswith('.java'):
            return_code = subprocess.call(['javac', src])
        elif src.endswith('.scala'):
            curdir = os.getcwd()  # scalac is a bit different when it comes to placing class files so we need to compensate.
            os.chdir('..')
            return_code = subprocess.call(['scalac', src])
            print '%s COMPILED!!!' % (src, )
            os.chdir(curdir)
        else:
            bad_value(source_files, "Can only handle java and scala files.")
        assert return_code == 0, 'Error while compiling "%s".  See message above.' % (src, )

    _compiled_srcs.add(source_files)


def jpype_connect_and_compile(root_dir=None, java_loc = None, source_files = []):
    """
    Open a JPype connection and compile the source files (in order).

    with jpype_connect_and_compile(source_files = ['./my_package/MyJavaClass.java'] as jp):
        c = jp.JClass('my_package.MyJavaClass')
        stuff = c.do_something()

    :param root_dir: The root of your java code (source files are referenced from here)
    :param java_loc: Location of Java on your system (None to guess (see _guess_java_loc above))
    :param source_files: A list of .java files to compile.
    :return: The connection object (use this in a "with" statement, as shown above)
    """
    connection = JPypeConnection(root_dir=root_dir, java_loc=java_loc)
    jpype_compile(source_files)
    return connection

