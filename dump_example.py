import sys
import lleaves
import lleaves.compiler.ast.dumpc as dumper
#import lleaves.compiler.ast.dump as dumper
#import lleaves.compiler.ast.dumpjs as dumper
print(
    dumper.format_dump(
        dumper.dump(
            lleaves.compiler.ast.parse_to_ast(sys.argv[1])
        )
    )
)
