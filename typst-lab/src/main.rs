use typst_syntax::*;

fn main() {
    // Create a string literal
    let code = r#"
= Hello

== Hello
    
Hi nice to meet you with #text(fill: blue)[qwq]
"#;
    let root = parse(code);
    if root.erroneous() {
        return;
    }

    println!("{:#?}", root);
}
