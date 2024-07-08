combine_postfits -i fitDiagnosticsTest.root --data --unblind --fit fit_s --style style.yml \
--rmap b220:r,m220:r --sigs b220,m220 --project-signals 5,5 --onto qcd --chi2 False \
--cats 'fail:ptbin*fail*;bb:ptbin*passTbvlpassL*;qq:ptbin*passTbvlfailL*'    \
--bkgs  'hbb,zbb,zqq,vv,wqq,top,vlep' --xlabel 'Jet $m_{SD}$' --ylabel 'Events / 5 GeV' \
--catlabels '$500 < p_T < 1200~GeV$\nParticleNet Fail;$500 < p_T < 1200~GeV$\n$ParticleNet_{2-prong}$ Pass\n$ParticleNet_{bb}$ Pass;$500 < p_T < 1200~GeV$\n$ParticleNet_{2-prong}$ Pass\n$ParticleNet_{bb}$ Fail' \
--cmslabel Preliminary  -vv
# --catlabels '$500 < p_T < 550$;$550 < p_T < 600$;$600 < p_T < 700$;$700 < p_T < 800$;$800 < p_T < 1200$'
