import json
import os


def scan_pandas_categorical(file_path):
    # Ich denke die Funktion waere leichter verstaendlich wenn du ein Beispiel
    # einbaust, nach was du genau suchst in der Datei.
    # Waere auch hilfreich zu zeigen, wie genau der Return value aussieht.
    pandas_key = "pandas_categorical:"
    offset = -len(pandas_key)
    max_offset = -os.path.getsize(file_path)
    # seek backwards from end of file until we have to lines
    # the (pen)ultimate line should be pandas_categorical:XXX
    with open(file_path, "rb") as f:
        while True:
            # Terminiert nicht wenn die Datei komplett Banane ist
            # und die break-Condition unten nie wahr wird.
            # -
            # Evtl. ist es effizienter und einfacher verstaendlich wenn man
            # den Loop so aendert, dass immer Bloecke fixer Groesse gelesen
            # werden (z.B. 10 KiB) und diese nach dem Key durchsucht werden.
            # Immer len(pandas_key) fortzuschreiten ist ziemlich spezifisch
            # und ich frage mich, ob das notwendig ist.
            # Allgemein ist es effizienter, wenn man weniger syscalls macht
            # (seek, read)
            if offset < max_offset:
                offset = max_offset
            f.seek(offset, os.SEEK_END)
            lines = f.readlines()
            if len(lines) >= 2:
                break
            offset *= 2
    # Hier waere nuetzlich eine Erklaerung oder ein Beispiel zu haben warum
    # es diese Faelle gibt.
    last_line = lines[-1].decode().strip()
    if not last_line.startswith(pandas_key):
        last_line = lines[-2].decode().strip()
    if last_line.startswith(pandas_key):
        return json.loads(last_line[len(pandas_key) :])
    raise ValueError("Ill formatted model file!")


def scan_model_file(file_path, general_info_only=False):
    res = {"trees": []}

    with open(file_path, "r") as f:
        # List of blocks we expect:
        # 1* General Information
        # N* Tree, one block for each tree
        # 1* 'end of trees'
        # 1* Feature importances
        # 1* Parameters
        # 1* 'end of parameters'
        # 1* 'pandas_categorical:XXXXX'
        lines = _get_next_block_of_lines(f)
        assert lines[0] == "tree" and lines[1].startswith(
            "version="
        ), f"{file_path} is not a LightGBM model file"
        res["general_info"] = _struct_from_block(lines, INPUT_SCAN_KEYS)
        if general_info_only:
            return res

        lines = _get_next_block_of_lines(f)
        while lines:
            if lines[0].startswith("Tree="):
                res["trees"].append(_scan_tree(lines))
            else:
                assert lines[0] == "end of trees"
                break
            lines = _get_next_block_of_lines(f)
    # Finde es ueberraschend, dass etwas mit dem Namen "pandas_categorical"
    # kein df ist, sondern irgend eine JSON-Struktur
    # Warum ist denn "Pandas" im Namen?
    res["pandas_categorical"] = scan_pandas_categorical(file_path)
    return res


def _scan_tree(lines):
    struct = _struct_from_block(lines, TREE_SCAN_KEYS)
    return struct


def _get_next_block_of_lines(file):
    # the only function where we advance file_offset
    # Was ist file_offset?
    result = []
    # Oder hier walrus operator nutzen :)
    line = file.readline()
    while line == "\n":
        line = file.readline()
    while line != "\n" and line != "":
        result.append(line.strip())
        line = file.readline()
    return result


def cat_args_bitmap(arr):
    # Feature infos for floats look like [x.xxxx:y.yyyy]
    # for categoricals like X:Y:Z:

    # Hier gefallen mir einige Dinge nicht:

    # 1. "cat_args" ist kein besonders aussagekraeftiger Name.
    # Koennte sowas wie "arg_is_categorical" besser passen?

    # 2. Ich finde es nicht als hilfreich, von einer Bitmap zu sprechen.
    # - "Bitmap" hat normalerweise eine sehr bestimmte Bedeutung
    # (low-level Datenstruktur zum Effizienz-optimierten Speichern + Lookup
    # von boolschen Werten per Index).
    # - es ist ein Detail, dass das eine Bitmap ist.
    # - man schreibt typischerweise den Typen nicht in den Namen,
    # z.B. "name_string" oder "lookup_dict".

    # 3. Die Funktion operiert auf einer Liste/Array, obwohl sie auf jedem Element
    # das gleiche macht. Sollte deshalb eine Funktion sein, die auf einem Element
    # arbeitet und dann per map() auf eine Liste angewendet werden.

    # 4. "arr" als Parametername ist nicht hilfreich.

    # Allgemein gefaellt mir das Herumreichen der Bitmap/Lookup-Liste im Code nicht besonders.
    # Ich wuerde lieber eine Methode auf dem Baum/Wald machen: "is_categorical_arg(x)"
    # dann unter der Haube evtl. eine Bitmap oder ein anderes Mapping nutzen kann.
    return [not val.startswith("[") for val in arr]


class ScannedValue:
    def __init__(self, type: type, is_list=False, null_ok=False):
        self.type = type
        self.is_list = is_list
        self.null_ok = null_ok


INPUT_SCAN_KEYS = {
    "max_feature_idx": ScannedValue(int),
    "version": ScannedValue(str),
    "feature_infos": ScannedValue(str, True),
    "objective": ScannedValue(str, True),
}
TREE_SCAN_KEYS = {
    "Tree": ScannedValue(int),
    "num_leaves": ScannedValue(int),
    "num_cat": ScannedValue(int),
    "split_feature": ScannedValue(int, True),
    "threshold": ScannedValue(float, True),
    "decision_type": ScannedValue(int, True),
    "left_child": ScannedValue(int, True),
    "right_child": ScannedValue(int, True),
    "leaf_value": ScannedValue(float, True),
    "cat_threshold": ScannedValue(int, True, True),
    "cat_boundaries": ScannedValue(int, True, True),
}


def _struct_from_block(lines: list, keys_to_scan: dict):
    """
    Scans a block (= list of lines), produces a key: value struct
    @param lines: list of lines in the block
    @param keys_to_scan: dict with 'key': 'type of value' of keys to scan for
    """
    # "struct_from_block" finde ich nicht so gut benannt, weil "struct" hier
    # wieder eine Typenbezeichnung ist und nicht wirklich aussagt, welche Bedeutung
    # dieses struct hat. Diese Funktion parsed ja einen Block in ein Map, deshalb
    # waere evtl. ein besserer Name einfach "parse_block"?
    struct = {}
    for line in lines:
        # initial line in file
        if line == "tree":
            continue

        key, value = line.split("=")
        value_type = keys_to_scan.get(key)
        if value_type is None:
            continue
        if value_type.is_list:
            if value:
                parsed_value = [value_type.type(x) for x in value.split(" ")]
            else:
                parsed_value = []
        else:
            parsed_value = value_type.type(value)
        struct[key] = parsed_value

    expected_keys = {k for k, v in keys_to_scan.items() if not v.null_ok}
    missing_keys = expected_keys - struct.keys()
    if missing_keys:
        raise Bla(f"Missing non-nullable keys {missing_keys}")
    return struct
