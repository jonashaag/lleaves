from decision_tree import Forest


def ir_from_json(json):
    return Forest(json).get_ir()
