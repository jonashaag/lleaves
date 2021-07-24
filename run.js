const forest_root = require("./js.js")
const F = 5
const n_in = 5000000//0
const in_ = []
for (let i = 0; i < n_in; ++i) {
  const inner = new Array(F)
  for (let j = 0; j < F; ++j) {
    inner[j] = Math.random() * 50
  }
  in_.push(inner)
}
const out = new Array(n_in)
const start = new Date()
forest_root(in_, out, 0, n_in)
console.log((new Date() - start) / 1000)
console.log(out[0])
