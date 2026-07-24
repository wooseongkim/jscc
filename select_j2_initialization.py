from __future__ import annotations
import argparse,json
from pathlib import Path
from speech_jscc.diagnostics.j2_barrage import select_initialization

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--fresh-summary",required=True);parser.add_argument("--transfer-summary",required=True);parser.add_argument("--output",required=True)
    args=parser.parse_args();fresh=json.loads(Path(args.fresh_summary).read_text());transfer=json.loads(Path(args.transfer_summary).read_text())
    decision=select_initialization(fresh["initialization_comparison_metrics"],transfer["initialization_comparison_metrics"])
    decision.update({"fresh_summary":args.fresh_summary,"transfer_summary":args.transfer_summary})
    Path(args.output).parent.mkdir(parents=True,exist_ok=True);Path(args.output).write_text(json.dumps(decision,indent=2));print(json.dumps(decision,indent=2))

if __name__=="__main__":main()
