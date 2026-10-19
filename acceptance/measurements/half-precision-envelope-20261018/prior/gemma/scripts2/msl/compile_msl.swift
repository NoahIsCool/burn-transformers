// Scratch helper (not committed): compiles MSL source files with the system Metal compiler and
// prints, per file, "OK" or the compiler's diagnostic. Usage: swift compile_msl.swift [--version 3.1] FILE...
import Foundation
import Metal

var args = Array(CommandLine.arguments.dropFirst())
var version: MTLLanguageVersion? = nil
if args.first == "--version", args.count > 1 {
    switch args[1] {
    case "3.0": version = .version3_0
    case "3.1": version = .version3_1
    case "3.2": version = .version3_2
    default: version = nil
    }
    args.removeFirst(2)
}
guard let device = MTLCreateSystemDefaultDevice() else { print("no Metal device"); exit(2) }
print("device: \(device.name)")
for path in args {
    let source = try! String(contentsOfFile: path, encoding: .utf8)
    let options = MTLCompileOptions()
    if let v = version { options.languageVersion = v }
    do {
        _ = try device.makeLibrary(source: source, options: options)
        print("\(path): OK")
    } catch {
        print("\(path): ERROR \(error.localizedDescription)")
    }
}
