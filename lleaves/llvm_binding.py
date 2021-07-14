from pathlib import Path

import llvmlite.binding as llvm


def compile_module_to_asm(module, cache_path=None):
    # fände immer noch interessant zu wissen ob es hier global state gibt mit llvm oder ob das thread-safe ist.
    llvm.initialize()
    llvm.initialize_native_target()
    llvm.initialize_native_asmprinter()

    # Create a target machine representing the host
    target = llvm.Target.from_default_triple()
    target_machine = target.create_target_machine()

    # Create execution engine for our module
    execution_engine = llvm.create_mcjit_compiler(module, target_machine)

    # when caching we dump the executable once the module finished compiling
    # we only ever have one module, hence we can ignore the 'llvm_module' parameter
    if cache_path:
        # ich würde die cases umdrehen, immer True case als erstes und False case in das else.
        # und die beiden cases komplett explizit hinschreiben, indem die initialisierung der
        # variablen komplett in die cases gezogen wird.
        # und ggf. die Callbacks als lambda schreiben. 
        if Path(cache).exists():
            load_from_cache = lambda llvm_module: Path(cache_path).read_bytes()
            save_to_cache = None
        else:
            load_from_cache = None
            save_to_cache = lambda llvm_module, buffer: Path(cache_path).write_bytes(buffer)

    # eine ebene einrücken (if cache:)?
    execution_engine.set_object_cache(
        notify_func=save_to_cache, getbuffer_func=load_from_cache
    )

    # oder geht es sogar so einfach?
    if cache_path:
        if Path(cache_path).exists():
            execution_engine.set_object_cache(getbuffer_func=lambda _: Path(cache_path).read_bytes())
        else:
            execution_engine.set_object_cache(notify_func=lambda _, buffer: Path(cache_path).write_bytes(buffer))

    # compile IR to ASM
    execution_engine.finalize_object()
    execution_engine.run_static_constructors()
    return execution_engine
