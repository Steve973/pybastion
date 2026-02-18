"""
Knowledge base for Python semantic analysis.

Contains known patterns for:
- Operations that can raise exceptions
- Boundary operations (system boundaries)
- Deterministic operations (never raise)
- Standard library modules
"""


# Operations that ALWAYS can raise specific exceptions
OPERATIONS_THAT_RAISE = {
    'int': {
        'exceptions': ['ValueError', 'TypeError'],
        'reason': 'Invalid string or type for conversion',
        'ei_count': 2  # success, exception
    },
    'float': {
        'exceptions': ['ValueError', 'TypeError'],
        'reason': 'Invalid string or type for conversion',
        'ei_count': 2
    },
    'bool': {
        'exceptions': ['TypeError'],
        'reason': 'Invalid type (rare, but possible)',
        'ei_count': 2
    },
    'complex': {
        'exceptions': ['ValueError', 'TypeError'],
        'reason': 'Invalid string or type for conversion',
        'ei_count': 2
    },
}


# String methods that can raise
STRING_METHODS_THAT_RAISE = {
    'index': {
        'exceptions': ['ValueError'],
        'reason': 'Substring not found',
        'ei_count': 2
    },
    'rindex': {
        'exceptions': ['ValueError'],
        'reason': 'Substring not found',
        'ei_count': 2
    },
}


# JSON operations
JSON_OPERATIONS = {
    'json.loads': {
        'exceptions': ['JSONDecodeError'],
        'reason': 'Malformed JSON',
        'ei_count': 2
    },
    'json.load': {
        'exceptions': ['JSONDecodeError'],
        'reason': 'Malformed JSON',
        'ei_count': 2
    },
}


# Operations that are ALWAYS deterministic (never raise)
DETERMINISTIC_OPERATIONS = {
    'str',  # str() never raises
    'list',  # list() never raises (from iterable)
    'dict',  # dict() never raises
    'tuple',  # tuple() never raises
    'set',  # set() never raises
    'frozenset',
    'bytes',
    'bytearray',
}


# Boundary operations - filesystem
FILESYSTEM_OPERATIONS = {
    'open': {
        'kind': 'filesystem',
        'operation': 'read/write',
        'can_raise': True,
        'exceptions': ['FileNotFoundError', 'PermissionError', 'OSError']
    },
    'pathlib.Path.resolve': {
        'kind': 'filesystem',
        'operation': 'read',
        'can_raise': True,
        'exceptions': ['OSError', 'RuntimeError']
    },
    'pathlib.Path.relative_to': {
        'kind': 'filesystem',
        'operation': 'read',
        'can_raise': True,
        'exceptions': ['ValueError', 'OSError']
    },
    'pathlib.Path.open': {
        'kind': 'filesystem',
        'operation': 'read/write',
        'can_raise': True,
        'exceptions': ['FileNotFoundError', 'PermissionError', 'OSError']
    },
    'pathlib.Path.read_text': {
        'kind': 'filesystem',
        'operation': 'read',
        'can_raise': True,
        'exceptions': ['FileNotFoundError', 'PermissionError', 'OSError']
    },
    'pathlib.Path.write_text': {
        'kind': 'filesystem',
        'operation': 'write',
        'can_raise': True,
        'exceptions': ['PermissionError', 'OSError']
    },
    'pathlib.Path.mkdir': {
        'kind': 'filesystem',
        'operation': 'write',
        'can_raise': True,
        'exceptions': ['FileExistsError', 'PermissionError', 'OSError']
    },
    'pathlib.Path.unlink': {
        'kind': 'filesystem',
        'operation': 'write',
        'can_raise': True,
        'exceptions': ['FileNotFoundError', 'PermissionError', 'OSError']
    },
    'os.remove': {
        'kind': 'filesystem',
        'operation': 'write',
        'can_raise': True,
        'exceptions': ['FileNotFoundError', 'PermissionError', 'OSError']
    },
    'os.mkdir': {
        'kind': 'filesystem',
        'operation': 'write',
        'can_raise': True,
        'exceptions': ['FileExistsError', 'PermissionError', 'OSError']
    },
    'zipfile.ZipFile': {
        'kind': 'filesystem',
        'operation': 'read/write',
        'can_raise': True,
        'exceptions': ['BadZipFile', 'FileNotFoundError', 'PermissionError', 'OSError']
    },
    'zipfile.ZipFile.read': {
        'kind': 'filesystem',
        'operation': 'read',
        'can_raise': True,
        'exceptions': ['KeyError', 'BadZipFile', 'OSError']
    },
    'zipfile.ZipFile.open': {
        'kind': 'filesystem',
        'operation': 'read',
        'can_raise': True,
        'exceptions': ['KeyError', 'BadZipFile', 'OSError']
    },
    'zipfile.ZipFile.extractall': {
        'kind': 'filesystem',
        'operation': 'write',
        'can_raise': True,
        'exceptions': ['BadZipFile', 'PermissionError', 'OSError']
    },
    'zipfile.ZipFile.namelist': {
        'kind': 'filesystem',
        'operation': 'read',
        'can_raise': False
    },
}


# Boundary operations - network (HTTP)
NETWORK_HTTP_OPERATIONS = {
    'requests.get': {
        'kind': 'network',
        'protocol': 'http',
        'operation': 'read',
        'can_raise': True,
        'exceptions': ['RequestException', 'ConnectionError', 'Timeout']
    },
    'requests.post': {
        'kind': 'network',
        'protocol': 'http',
        'operation': 'write',
        'can_raise': True,
        'exceptions': ['RequestException', 'ConnectionError', 'Timeout']
    },
    'requests.put': {
        'kind': 'network',
        'protocol': 'http',
        'operation': 'write',
        'can_raise': True,
        'exceptions': ['RequestException', 'ConnectionError', 'Timeout']
    },
    'requests.delete': {
        'kind': 'network',
        'protocol': 'http',
        'operation': 'write',
        'can_raise': True,
        'exceptions': ['RequestException', 'ConnectionError', 'Timeout']
    },
    'requests.head': {
        'kind': 'network',
        'protocol': 'http',
        'operation': 'read',
        'can_raise': True,
        'exceptions': ['RequestException', 'ConnectionError', 'Timeout']
    },
    'urllib.request.urlopen': {
        'kind': 'network',
        'protocol': 'http',
        'can_raise': True,
        'exceptions': ['URLError', 'HTTPError']
    },
}


# Boundary operations - database
DATABASE_OPERATIONS = {
    'cursor.execute': {
        'kind': 'database',
        'operation': 'query',
        'can_raise': True,
        'exceptions': ['DatabaseError', 'IntegrityError']
    },
    'cursor.executemany': {
        'kind': 'database',
        'operation': 'query',
        'can_raise': True,
        'exceptions': ['DatabaseError', 'IntegrityError']
    },
    'sqlite3.connect': {
        'kind': 'database',
        'operation': 'connect',
        'can_raise': True,
        'exceptions': ['DatabaseError']
    },
}


# Boundary operations - environment
ENVIRONMENT_OPERATIONS = {
    'os.getenv': {
        'kind': 'env',
        'operation': 'read',
        'can_raise': False  # Returns None if not found
    },
    'os.environ': {
        'kind': 'env',
        'operation': 'read',
        'can_raise': True,  # Can raise KeyError
        'exceptions': ['KeyError']
    },
}


# Boundary operations - clock/time
CLOCK_OPERATIONS = {
    'datetime.now': {
        'kind': 'clock',
        'operation': 'read',
        'can_raise': False
    },
    'datetime.utcnow': {
        'kind': 'clock',
        'operation': 'read',
        'can_raise': False
    },
    'datetime.today': {
        'kind': 'clock',
        'operation': 'read',
        'can_raise': False
    },
    'time.time': {
        'kind': 'clock',
        'operation': 'read',
        'can_raise': False
    },
    'time.monotonic': {
        'kind': 'clock',
        'operation': 'read',
        'can_raise': False
    },
    'time.perf_counter': {
        'kind': 'clock',
        'operation': 'read',
        'can_raise': False
    },
}


# Boundary operations - randomness
RANDOMNESS_OPERATIONS = {
    'random.random': {
        'kind': 'randomness',
        'operation': 'generate',
        'can_raise': False
    },
    'random.randint': {
        'kind': 'randomness',
        'operation': 'generate',
        'can_raise': False
    },
    'random.choice': {
        'kind': 'randomness',
        'operation': 'generate',
        'can_raise': True,
        'exceptions': ['IndexError']  # If sequence is empty
    },
    'random.shuffle': {
        'kind': 'randomness',
        'operation': 'generate',
        'can_raise': False
    },
}


# Boundary operations - subprocess
SUBPROCESS_OPERATIONS = {
    'subprocess.run': {
        'kind': 'subprocess',
        'operation': 'execute',
        'can_raise': True,
        'exceptions': ['CalledProcessError', 'TimeoutExpired', 'OSError']
    },
    'subprocess.call': {
        'kind': 'subprocess',
        'operation': 'execute',
        'can_raise': True,
        'exceptions': ['OSError']
    },
    'subprocess.Popen': {
        'kind': 'subprocess',
        'operation': 'execute',
        'can_raise': True,
        'exceptions': ['OSError']
    },
}


# Combine all boundary operations
BOUNDARY_OPERATIONS = {
    **FILESYSTEM_OPERATIONS,
    **NETWORK_HTTP_OPERATIONS,
    **DATABASE_OPERATIONS,
    **ENVIRONMENT_OPERATIONS,
    **CLOCK_OPERATIONS,
    **RANDOMNESS_OPERATIONS,
    **SUBPROCESS_OPERATIONS,
}

STDLIB_CLASSES = {
    # Built-in types (not technically stdlib but treated as such for classification)
    'str', 'int', 'float', 'bool', 'bytes', 'bytearray',
    'list', 'tuple', 'dict', 'set', 'frozenset',
    'range', 'enumerate', 'zip', 'filter', 'map',

    # pathlib
    'Path', 'PurePath', 'PosixPath', 'WindowsPath', 'PurePosixPath', 'PureWindowsPath',

    # datetime
    'datetime', 'date', 'time', 'timedelta', 'timezone', 'tzinfo',

    # decimal/fractions
    'Decimal', 'Context',
    'Fraction',

    # enum
    'Enum', 'IntEnum', 'Flag', 'IntFlag', 'StrEnum',

    # collections
    'OrderedDict', 'defaultdict', 'Counter', 'deque', 'ChainMap',
    'UserDict', 'UserList', 'UserString',

    # re
    'Pattern', 'Match',

    # dataclasses
    'dataclass', 'field',

    # typing (commonly imported as names)
    'NamedTuple', 'TypedDict',

    # io
    'StringIO', 'BytesIO', 'TextIOWrapper', 'BufferedReader', 'BufferedWriter',

    # threading
    'Thread', 'Lock', 'RLock', 'Semaphore', 'BoundedSemaphore', 'Event',
    'Condition', 'Timer', 'Barrier',

    # queue
    'Queue', 'LifoQueue', 'PriorityQueue', 'SimpleQueue',

    # compression
    'ZipFile', 'ZipExtFile', 'TarFile', 'TarInfo', 'gzip', 'bz2', 'lzma'

    # argparse
    'ArgumentParser', 'Namespace',

    # logging
    'Logger', 'Handler', 'Formatter', 'Filter', 'LogRecord',

    # subprocess
    'Popen', 'CompletedProcess', 'TimeoutExpired',

    # socket
    'socket',  # the socket class itself

    # http
    'HTTPStatus', 'HTTPResponse', 'HTTPConnection',

    # email
    'EmailMessage', 'Message',

    # xml.etree.ElementTree
    'Element', 'ElementTree',

    # json
    'JSONEncoder', 'JSONDecoder',

    # abc
    'ABC', 'ABCMeta',

    # contextlib
    'contextmanager', 'asynccontextmanager', 'suppress', 'ExitStack', 'AsyncExitStack',

    # functools
    'partial', 'partialmethod', 'cached_property',

    # tempfile
    'TemporaryFile', 'NamedTemporaryFile', 'SpooledTemporaryFile', 'TemporaryDirectory',

    # weakref
    'WeakValueDictionary', 'WeakKeyDictionary', 'WeakSet', 'WeakMethod',
    'ref', 'proxy', 'finalize',

    # concurrent.futures
    'ThreadPoolExecutor', 'ProcessPoolExecutor', 'Future', 'Executor',

    # multiprocessing
    'Process', 'Pool', 'Manager', 'Value', 'Array',

    # asyncio
    'Task', 'Future',  # asyncio versions (namespace collision with concurrent.futures)

    # unittest
    'TestCase', 'TestSuite', 'TestLoader', 'TestResult',

    # types (commonly used)
    'SimpleNamespace', 'MappingProxyType',
}


# Python standard library modules
STDLIB_MODULES = {
    # Core
    'sys', 'os', 'io', 'pathlib', 'typing', 'types', 'builtins',

    # Data structures
    'collections', 'array', 'heapq', 'bisect', 'queue', 'weakref', 'copy',

    # Iteration & functional
    'itertools', 'functools', 'operator',

    # Text processing
    're', 'difflib', 'textwrap', 'unicodedata', 'stringprep', 'string',

    # Binary data
    'struct', 'codecs',

    # Date/Time
    'datetime', 'time', 'calendar', 'zoneinfo',

    # Math
    'math', 'cmath', 'decimal', 'fractions', 'random', 'statistics',

    # File/Directory
    'fileinput', 'tempfile', 'glob', 'fnmatch', 'shutil',

    # Data persistence
    'pickle', 'shelve', 'dbm', 'sqlite3',

    # Compression
    'zlib', 'gzip', 'bz2', 'lzma', 'zipfile', 'tarfile',

    # Formats
    'csv', 'json', 'xml', 'html', 'configparser', 'tomllib', 'plistlib',

    # Crypto
    'hashlib', 'hmac', 'secrets',

    # OS interface
    'subprocess', 'signal', 'errno', 'ctypes',

    # Concurrency
    'threading', 'multiprocessing', 'concurrent', 'asyncio', 'contextvars',

    # Networking
    'socket', 'ssl', 'select', 'selectors',

    # Internet protocols
    'email', 'mailbox', 'mimetypes', 'base64', 'binascii', 'quopri', 'uu',

    # HTTP/URL
    'http', 'urllib',

    # Logging
    'logging',

    # Testing
    'unittest', 'doctest', 'mock',

    # Debugging/introspection
    'pdb', 'trace', 'traceback', 'warnings', 'inspect', 'dis',

    # Runtime services
    'abc', 'atexit', 'enum', 'dataclasses', 'contextlib', 'importlib',
    'pkgutil', 'modulefinder', 'runpy', 'site', 'sysconfig',
}


PYTHON_BUILTINS = {
    # Built-in functions
    'abs', 'all', 'any', 'ascii', 'bin', 'bool', 'breakpoint', 'bytearray',
    'bytes', 'callable', 'chr', 'classmethod', 'compile', 'complex',
    'delattr', 'dict', 'dir', 'divmod', 'enumerate', 'eval', 'exec',
    'filter', 'float', 'format', 'frozenset', 'getattr', 'globals',
    'hasattr', 'hash', 'help', 'hex', 'id', 'input', 'int', 'isinstance',
    'issubclass', 'iter', 'len', 'list', 'locals', 'map', 'max',
    'memoryview', 'min', 'next', 'object', 'oct', 'open', 'ord', 'pow',
    'print', 'property', 'range', 'repr', 'reversed', 'round', 'set',
    'setattr', 'slice', 'sorted', 'staticmethod', 'str', 'sum', 'super',
    'tuple', 'type', 'vars', 'zip',

    # Built-in exceptions
    'BaseException', 'Exception', 'ArithmeticError', 'AssertionError',
    'AttributeError', 'BlockingIOError', 'BrokenPipeError', 'BufferError',
    'ChildProcessError', 'ConnectionError', 'EOFError', 'FileExistsError',
    'FileNotFoundError', 'FloatingPointError', 'ImportError', 'IndexError',
    'InterruptedError', 'IsADirectoryError', 'KeyError', 'KeyboardInterrupt',
    'LookupError', 'MemoryError', 'NameError', 'NotADirectoryError',
    'NotImplementedError', 'OSError', 'OverflowError', 'PermissionError',
    'ProcessLookupError', 'RecursionError', 'ReferenceError', 'RuntimeError',
    'StopIteration', 'StopAsyncIteration', 'SyntaxError', 'SystemError',
    'SystemExit', 'TabError', 'TimeoutError', 'TypeError', 'UnboundLocalError',
    'UnicodeError', 'UnicodeDecodeError', 'UnicodeEncodeError',
    'UnicodeTranslateError', 'ValueError', 'ZeroDivisionError',

    # Typing utilities (from typing module but commonly used bare)
    'cast', 'Any', 'Union', 'Optional', 'List', 'Dict', 'Set', 'Tuple',
    'Callable', 'TypeVar', 'Generic', 'Protocol',

    # Special
    'cls', 'self',  # Not actually builtins but should never be integrations
}


COMMON_EXTLIB_MODULES = {
    'aiohttp',
    'bs4',
    'certifi',
    'cffi',
    'chardet',
    'cryptography',
    'httpx',
    'idna',
    'itsdangerous',
    'jinja2',
    'lxml',
    'markupsafe',
    'numpy',
    'packaging',
    'pandas',
    'pyOpenSSL',
    'pyasn1',
    'pyasn1_modules',
    'pycparser',
    'pydantic',
    'pygments',
    'pyjwt',
    'pyopenssl',
    'pyparsing',
    'pysocks',
    'pytz',
    'pyyaml',
    'requests',
    'resolvelib',
    'setuptools',
    'tomli',
    'tomli_w',
    'urllib3',
    'werkzeug',
    'yaml',
}


BUILTIN_METHODS = {
    # Known builtin method patterns (never integrations)
    'items', 'keys', 'values',           # dict methods
    'get', 'setdefault', 'update',       # dict methods
    'append', 'extend', 'pop',           # list methods
    'add', 'remove', 'discard',          # set methods
    'split', 'join', 'strip',            # str methods
}


def get_operation_info(target: str) -> dict | None:
    """
    Get information about an operation from the knowledge base.

    Returns dict with operation details, or None if not in knowledge base.
    """
    # Check if it's a known operation that raises
    if target in OPERATIONS_THAT_RAISE:
        return {
            'category': 'type_conversion',
            **OPERATIONS_THAT_RAISE[target],
            'source': 'knowledge_base'
        }

    # Check string methods
    if '.' in target:
        method = target.split('.')[-1].split('(')[0]
        if method in STRING_METHODS_THAT_RAISE:
            return {
                'category': 'string_method',
                **STRING_METHODS_THAT_RAISE[method],
                'source': 'knowledge_base'
            }

    # Check JSON operations
    if target in JSON_OPERATIONS:
        return {
            'category': 'json',
            **JSON_OPERATIONS[target],
            'source': 'knowledge_base'
        }

    # Check boundary operations
    if target in BOUNDARY_OPERATIONS:
        return {
            'category': 'boundary',
            **BOUNDARY_OPERATIONS[target],
            'source': 'knowledge_base'
        }

    # Check if deterministic
    if target in DETERMINISTIC_OPERATIONS:
        return {
            'category': 'deterministic',
            'can_raise': False,
            'ei_count': 1,
            'source': 'knowledge_base'
        }

    return None


def is_stdlib_module(module_name: str) -> bool:
    """Check if a module is part of Python's standard library."""
    # Get the root module name
    root = module_name.split('.')[0]
    return root in STDLIB_MODULES