import lleaves
import lleaves.compiler.ast.dumpc as dumper
#import lleaves.compiler.ast.dump as dumper
print(
    dumper.format_dump(
        dumper.dump(
            lleaves.compiler.ast.parse_to_ast("tests/models/NYC_taxi/model.txt")
        )
    )
)
