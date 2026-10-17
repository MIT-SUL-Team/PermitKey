import unittest,json,tempfile
from pathlib import Path
from permitkey_release.aggregate import aggregate
from permitkey_release.common import MASK,HEADS
from permitkey_release.validation import summarize
from permitkey_release.evidence_policy import assess

ROOT=Path(__file__).resolve().parents[1]

class ReleaseTests(unittest.TestCase):
    def test_human_metrics(self):
        r=summarize(ROOT/'validation/human_review_metrics.json')
        self.assertAlmostEqual(r['micro_f1'],.7991746905089408)
        self.assertAlmostEqual(r['macro_f1'],.7916827679005705)
        self.assertEqual((r['tp'],r['fp'],r['fn']),(581,134,158))

    def test_partition_and_multilabel_counts(self):
        import duckdb
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'tract'
            r=aggregate(ROOT/'examples/canonical_records.jsonl',out,'test')
            self.assertEqual((r['input'],r['eligible'],r['positive14']),(7,3,1))
            self.assertEqual(sum(r[k] for k in ['excluded_by_keep','provisional_excluded','missing_or_invalid_issue_year','unresolved_target_geography','eligible']),7)
            c=duckdb.connect()
            data=c.execute('SELECT n_records,n_uncategorized,cnt_SOLAR_PV,cnt_ROOFING FROM read_parquet(?)',[str(out/'tract_year.parquet')]).fetchall()
            self.assertEqual(data,[(3,2,1,1)])
            c.close()

    def test_jurisdiction_retains_unlocated(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=aggregate(ROOT/'examples/canonical_records.jsonl',Path(tmp)/'jurisdiction','test','jurisdiction')
            self.assertEqual((r['eligible'],r['positive14']),(4,2))

    def test_reject_duplicate_identity(self):
        r=json.loads((ROOT/'examples/canonical_records.jsonl').read_text().splitlines()[0])
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'bad.jsonl';p.write_text(json.dumps(r)+'\n'+json.dumps(r)+'\n')
            with self.assertRaisesRegex(ValueError,'stable_row_key'):
                aggregate(p,Path(tmp)/'out','test')

    def test_reject_missing_source_identity(self):
        r=json.loads((ROOT/'examples/canonical_records.jsonl').read_text().splitlines()[0]);r['dataset_id']=None
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'bad.jsonl';p.write_text(json.dumps(r)+'\n')
            with self.assertRaisesRegex(ValueError,'source'):
                aggregate(p,Path(tmp)/'out','test')

    def test_refuse_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileExistsError):
                aggregate(ROOT/'examples/canonical_records.jsonl',tmp,'test')

    def test_preserved_evidence_policy(self):
        self.assertEqual(assess('REROOF')[2],1<<HEADS.index('ROOFING'))
        self.assertEqual(assess('no battery storage')[1],0)
        self.assertEqual(assess('')[3],'empty')

    def test_generic_label_rule(self):
        from permitkey_release import generic_label_rule as g
        p={'SOLAR_PV':1,'EV_CHARGER':1,'ROOFING':1}
        self.assertEqual(g.apply({'description':'Electrical Permit','native_type':'','native_subtype':''},p),{'SOLAR_PV':0,'EV_CHARGER':0,'ROOFING':1})
        self.assertEqual(g.apply({'description':'Install rooftop photovoltaic panels'},p),p)

if __name__=='__main__':unittest.main()
