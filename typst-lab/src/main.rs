use pretty::{BoxDoc, Doc, RcDoc};
use typst_syntax::*;

use SExp::*;
enum SExp {
    Atomic(u32),
    List(Vec<SExp>),
}

impl SExp {
    pub fn to_doc(&self) -> RcDoc<()> {
        match self {
            Atomic(n) => RcDoc::as_string(n),
            List(xs) => RcDoc::text("(")
                .append(RcDoc::intersperse(
                    xs.into_iter().map(|x| x.to_doc()),
                    Doc::line(),
                ))
                .nest(4)
                .group()
                .append(RcDoc::text(")")),
        }
    }

    pub fn to_pretty(&self, width: usize) -> String {
        let mut w = Vec::new();
        self.to_doc().render(width, &mut w).unwrap();
        String::from_utf8(w).unwrap()
    }
}

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

    let e = List(vec![
        Atomic(1),
        Atomic(2),
        List(vec![Atomic(3), Atomic(4), List(vec![Atomic(5), Atomic(6)])]),
    ]);
    println!("{}", e.to_pretty(12));
}
