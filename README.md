- Depends on combine, rhalphalib
- Run like 
```
python jeff_rhalpha.py --opath dummyx/ --ipt 2 --irho 2 --MCTF --irhoMC 2 --iptMC 2 --tagger pnmd2prong_0p05 --root_file TEMPLATES_blind2reg.root --year 2017 --tworeg -vv
```

Run postfit plots like 
```
combine_postfits -i fitDiagnosticsTest.root --data --style style.yml --sigs b150,m150 --project-signals 3,3 --rmap '{"m150": "r_q", "b150":"r_b"}' --bkgs top,vlep,wqq,zqq,zbb,hbb --onto 2017_qcd -p
```