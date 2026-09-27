import csv
import tempfile
import unittest
from pathlib import Path
from pipeline import Record, Index, entity_f05, record, features, FEATURE_NAMES, read_tsv
from validate_outputs import validate


class PipelineTests(unittest.TestCase):
    def row(self, eid, name, address, country='US'):
        return record({'entity_id':eid,'business_name':name,'business_address':address,'country':country})

    def test_metric_singletons_and_false_merges(self):
        self.assertEqual(entity_f05([],[]),1)
        self.assertEqual(entity_f05([],['S2-a']),0)
        self.assertEqual(entity_f05(['S2-a'],[]),0)
        self.assertAlmostEqual(entity_f05(['a','b'],['a','b','c']),5/7)

    def test_open_country_and_accents(self):
        a=self.row('S1-a','Café Étoile SARL','12 rue de Paris','France')
        b=self.row('S2-a','Cafe Etoile','12 rue de Paris','France')
        self.assertEqual(list(Index([a]).query_batch([b])),[[0]])
        self.assertEqual(a.name,b.name)

    def test_abbreviations(self):
        a=self.row('S1-a','Acme Corporation','123 Main St')
        b=self.row('S2-a','Acme Corp','123 Main Street')
        self.assertEqual(a.name,b.name)
        self.assertEqual(a.address,b.address)
        self.assertEqual(len(features(a,b)),len(FEATURE_NAMES))

    def test_blank_records_not_everything_matches(self):
        a=self.row('S1-a','','')
        b=self.row('S2-a','','')
        self.assertEqual(list(Index([a]).query_batch([b])),[[]])
        self.assertEqual(features(a,b)[0],0)

    def test_country_boundary(self):
        a=self.row('S1-a','Acme','12 Main St','US')
        b=self.row('S2-a','Acme','12 Main St','France')
        self.assertEqual(list(Index([a]).query_batch([b])),[[]])

    def test_many_matches_and_duplicate_block_keys(self):
        a=self.row('S1-a','Acme','12 Main St')
        b=self.row('S2-a','Acme','12 Main Street')
        c=self.row('S3-b','ACME','12 Main St')
        self.assertEqual(list(Index([a]).query_batch([b,c])),[[0],[0]])

    def test_oversized_block_is_skipped(self):
        rows=[self.row(f'S1-{i}','Acme','12 Main St') for i in range(3)]
        self.assertEqual(list(Index(rows,max_block=2).query_batch([rows[0]])),[[]])

    def test_sampling_preserves_full_block_suppression(self):
        import numpy as np
        rows=[self.row(f'S1-{i}','Acme','12 Main St') for i in range(3)]
        index=Index(rows,max_block=2)
        index.select_references(np.array([True,False,False]))
        self.assertEqual(list(index.query_batch([rows[0]])),[[]])

    def test_empty_keys_preserve_batch_ownership(self):
        a=self.row('S1-a','Acme','12 Main St')
        blank=self.row('S2-blank','','')
        self.assertEqual(list(Index([a]).query_batch([blank,a,blank,a])),[[],[0],[],[0]])

    def test_id_does_not_change_features(self):
        a=self.row('S1-1','Acme','12 Main St');b=self.row('S2-2','Acme','12 Main St')
        before=features(a,b);a.id='S1-999';b.id='S3-888'
        self.assertEqual(before,features(a,b))

    def test_tsv_round_trip_empty_comma_address(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'sample.tsv'
            with p.open('w',newline='',encoding='utf-8') as f:
                writer=csv.writer(f,delimiter='\t');writer.writerow(['id','value']);writer.writerow(['S1-a','']);writer.writerow(['S1-b','12 Main St, Apt 3'])
            self.assertEqual(list(read_tsv(p)),[{'id':'S1-a','value':''},{'id':'S1-b','value':'12 Main St, Apt 3'}])

    def output_fixture(self,tmp):
        p=Path(tmp);data=p/'data';out=p/'output';data.mkdir();out.mkdir()
        (data/'test_source1.tsv').write_text('entity_id\tbusiness_name\tbusiness_address\tcountry\nS1-a\tAcme\t12 Main\tUS\nS1-b\tNo match\t\tFrance\n')
        for source in (2,3):
            (data/f'test_source{source}.tsv').write_text(f'entity_id\tbusiness_name\tbusiness_address\tcountry\nS{source}-a\tAcme\t12 Main\tUS\n')
        (out/'matching_results.tsv').write_text('source1_entity_id\tmatched_entity_ids\nS1-a\tS2-a,S3-a\nS1-b\t\n')
        (out/'candidate_pairs.tsv').write_text('source1_entity_id\tcandidate_entity_ids\nS1-a\tS2-a,S3-a\nS1-b\t\n')
        return data,out

    def test_output_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            data,out=self.output_fixture(tmp)
            result=validate(data,out)
            self.assertEqual(result['source1_rows'],2)
            self.assertEqual(result['matched_pairs'],2)
            self.assertEqual(result['empty_predictions'],1)

    def test_validator_rejects_match_outside_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            data,out=self.output_fixture(tmp)
            (out/'candidate_pairs.tsv').write_text('source1_entity_id\tcandidate_entity_ids\nS1-a\tS2-a\nS1-b\t\n')
            with self.assertRaisesRegex(ValueError,'outside candidates'):validate(data,out)

    def test_validator_rejects_unknown_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            data,out=self.output_fixture(tmp)
            (out/'candidate_pairs.tsv').write_text('source1_entity_id\tcandidate_entity_ids\nS1-a\tS2-a,S3-a,S2-unknown\nS1-b\t\n')
            with self.assertRaisesRegex(ValueError,'Unknown candidate'):validate(data,out)

    def test_validator_rejects_missing_source1(self):
        with tempfile.TemporaryDirectory() as tmp:
            data,out=self.output_fixture(tmp)
            (out/'matching_results.tsv').write_text('source1_entity_id\tmatched_entity_ids\nS1-a\tS2-a,S3-a\n')
            with self.assertRaisesRegex(ValueError,'row counts'):validate(data,out)

    def test_validator_rejects_duplicate_id_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            data,out=self.output_fixture(tmp)
            (out/'matching_results.tsv').write_text('source1_entity_id\tmatched_entity_ids\nS1-a\tS2-a,S2-a\nS1-b\t\n')
            with self.assertRaisesRegex(ValueError,'Duplicate list'):validate(data,out)

if __name__=='__main__': unittest.main()
