import ctypes
import os
import sys


def main() -> int:
    if len(sys.argv) != 4:
        print(
            "Usage: winprop_convert_oda_to_odb.py <source-without-extension> <dest-without-extension> <winprop-api-bin>",
            file=sys.stderr,
        )
        return 2

    source = os.path.abspath(sys.argv[1])
    dest = os.path.abspath(sys.argv[2])
    winprop_api_bin = os.path.abspath(sys.argv[3])

    os.environ["RADFLEX_PATH"] = winprop_api_bin
    os.add_dll_directory(winprop_api_bin)

    callback_progress = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_int, ctypes.c_char_p)
    callback_message = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_char_p)
    callback_error = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_char_p, ctypes.c_int)

    class WinPropCallback(ctypes.Structure):
        _pack_ = 8
        _fields_ = [
            ("Percentage", callback_progress),
            ("Message", callback_message),
            ("Error", callback_error),
        ]

    class WinPropConverter(ctypes.Structure):
        _pack_ = 8
        _fields_ = [
            ("ConverterID", ctypes.c_int),
            ("databaseNameSource", ctypes.c_char_p),
            ("databaseNameDest", ctypes.c_char_p),
            ("measurementUnit", ctypes.c_char_p),
            ("ignoreList", ctypes.c_void_p),
            ("nbrElements", ctypes.c_int),
            ("scalingFactor", ctypes.c_float),
        ]

    def on_progress(progress, text):
        if text:
            print(f"{progress}% {text.decode('mbcs', 'replace')}")
        return 0

    def on_message(text):
        if text:
            print(text.decode("mbcs", "replace"))
        return 0

    def on_error(text, number):
        message = text.decode("mbcs", "replace") if text else ""
        print(f"Error ({number}): {message}", file=sys.stderr)
        return 0

    progress_cb = callback_progress(on_progress)
    message_cb = callback_message(on_message)
    error_cb = callback_error(on_error)
    callback = WinPropCallback(progress_cb, message_cb, error_cb)

    engine = ctypes.WinDLL(os.path.join(winprop_api_bin, "Engine.dll"))
    engine.WinProp_Structure_Init_Converter.argtypes = [ctypes.POINTER(WinPropConverter)]
    engine.WinProp_Structure_Init_Converter.restype = None
    engine.WinProp_Convert.argtypes = [
        ctypes.POINTER(WinPropConverter),
        ctypes.POINTER(WinPropCallback),
    ]
    engine.WinProp_Convert.restype = ctypes.c_int

    converter = WinPropConverter()
    engine.WinProp_Structure_Init_Converter(ctypes.byref(converter))
    converter.ConverterID = 201
    converter.databaseNameSource = source.encode("mbcs")
    converter.databaseNameDest = dest.encode("mbcs")
    converter.measurementUnit = b"Meter"
    converter.ignoreList = None
    converter.nbrElements = 0
    converter.scalingFactor = 1.0

    rc = engine.WinProp_Convert(ctypes.byref(converter), ctypes.byref(callback))
    if rc != 0:
        print(f"WinProp_Convert failed with code {rc}", file=sys.stderr)
        return rc

    output = dest + ".odb"
    if not os.path.exists(output):
        print(f"Conversion returned success, but output was not created: {output}", file=sys.stderr)
        return 3

    print(f"Created {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
