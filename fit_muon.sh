idir=$1


cd $idir
combineCards.py muonfail=`ls muonCRfailT.txt` muonpass=`ls muonCRpassT*.txt` > muon_combined.txt
sed -i 's/4         5/4         -5/g' muon_combined.txt
text2workspace.py muon_combined.txt --PO 'map:*/*qcd:r[1,-5,5]' -o muon_combined.root
combine -M FitDiagnostics -d muon_combined.root --robustFit=1 --robustHesse=1 --saveShapes
root -l fitDiagnosticsTest.root << EOF
fit_b->Print()
.q
EOF
cd -
