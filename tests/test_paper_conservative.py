import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import pandas as pd
from simulation.etf_universe.daily import run_day
from simulation.etf_universe.account import PaperAccount
from simulation.etf_universe.config import PaperConfig

class ConservativePaperTests(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup); self.root=Path(tmp.name)
        days=pd.bdate_range('2026-08-03',periods=65)
        self.t=days[-2].date().isoformat(); self.next=days[-1].date().isoformat()
        self.frame=pd.DataFrame({'date':days,'open':1.0,'high':1.02,'low':.98,'close':1.0,'volume':10000})
        self.data={'159001':self.frame}
    def lanes(self,action):
        return {'trend':pd.DataFrame([{'symbol':'159001','name':'测试ETF','action':action,'action_reason':'趋势成立','validated_trend_strategies':'趋势动量;量价','validated_family_votes':2,'analysis_rank':1}]),'rebound':pd.DataFrame(),'defense':pd.DataFrame()}
    def test_only_formal_entry(self):
        a=run_day('conservative',self.t,self.lanes('继续观察'),self.data,self.root)
        self.assertEqual([],a['pending_orders'])
        self.assertEqual([],run_day('conservative',self.t,self.lanes('可以关注买入'),self.data,self.root)['pending_orders'])
    def test_t1_and_idempotence(self):
        a=run_day('conservative',self.t,self.lanes('可以关注买入'),self.data,self.root)
        self.assertEqual(0,len(a['trade_log'])); self.assertEqual(1,len(a['pending_orders']))
        a=run_day('conservative',self.next,self.lanes('继续观察'),self.data,self.root)
        self.assertEqual(1,len(a['trade_log'])); self.assertEqual(self.next,a['trade_log'][0]['date'])
        self.assertGreater(a['trade_log'][0]['commission'],0)
        self.assertGreater(a['trade_log'][0]['slippage_cost'],0)
        self.assertEqual(a,run_day('conservative',self.next,self.lanes('继续观察'),self.data,self.root))
    def test_sell_pending(self):
        run_day('conservative',self.t,self.lanes('可以关注买入'),self.data,self.root)
        with patch('simulation.etf_universe.daily.current_holding_advice',return_value=('趋势破坏退出','买入理由失效')):
            a=run_day('conservative',self.next,self.lanes('继续观察'),self.data,self.root)
        self.assertEqual('sell',a['pending_orders'][0]['side'])
        self.assertEqual(1,len(a['trade_log']))
        d3=(pd.Timestamp(self.next)+pd.offsets.BDay(1)).date().isoformat()
        extra=pd.DataFrame([{'date':d3,'open':1.0,'high':1.02,'low':.98,'close':1.0,'volume':10000}])
        a=run_day('conservative',d3,self.lanes('继续观察'),{'159001':pd.concat([self.frame,extra],ignore_index=True)},self.root)
        self.assertEqual(2,len(a['trade_log'])); self.assertEqual({},a['positions'])
    def test_isolation(self):
        a=run_day('conservative',self.t,self.lanes('可以关注买入'),self.data,self.root)
        b=PaperAccount(self.root,'progressive',PaperConfig()).load_or_create(self.t)
        self.assertNotEqual(a['strategy_version'],b['strategy_version']); self.assertEqual([],b['pending_orders'])
    def test_real_next_open_not_signal_close(self):
        run_day('conservative',self.t,self.lanes('可以关注买入'),self.data,self.root)
        changed=self.frame.copy()
        changed.loc[changed.date.eq(pd.Timestamp(self.next)),'open']=1.04
        changed.loc[changed.date.eq(pd.Timestamp(self.next)),'close']=1.50
        a=run_day('conservative',self.next,self.lanes('继续观察'),{'159001':changed},self.root)
        self.assertAlmostEqual(1.04 * (1 + PaperConfig().slippage),a['trade_log'][0]['price'],places=4)
        self.assertEqual(2,len(a['nav_history']))
        self.assertEqual(a,run_day('conservative',self.next,self.lanes('继续观察'),{'159001':changed},self.root))
    def test_real_holdings_file_is_never_read(self):
        with patch('etf_universe.holdings.load_holdings',side_effect=AssertionError('real holdings read')):
            a=run_day('conservative',self.t,self.lanes('继续观察'),self.data,self.root)
        self.assertEqual({},a['positions'])
    def test_limit_up_keeps_order_unfilled(self):
        run_day('conservative',self.t,self.lanes('可以关注买入'),self.data,self.root)
        changed=self.frame.copy()
        changed.loc[changed.date.eq(pd.Timestamp(self.next)),'open']=1.10
        a=run_day('conservative',self.next,self.lanes('继续观察'),{'159001':changed},self.root)
        self.assertEqual([],a['trade_log'])
        self.assertEqual({},a['positions'])
    def test_pending_recovers_after_reload(self):
        first=run_day('conservative',self.t,self.lanes('可以关注买入'),self.data,self.root)
        self.assertEqual(first,PaperAccount(self.root,'conservative',PaperConfig()).load_or_create(self.t))
        second=run_day('conservative',self.next,self.lanes('继续观察'),self.data,self.root)
        self.assertIn(first['pending_orders'][0]['id'],second['executed_order_ids'])
    def test_corrupt_state_is_not_overwritten(self):
        path=PaperAccount(self.root,'conservative',PaperConfig()).path
        path.write_text('{broken',encoding='utf-8')
        with self.assertRaises(ValueError):
            run_day('conservative',self.t,self.lanes('可以关注买入'),self.data,self.root)
        self.assertEqual('{broken',path.read_text(encoding='utf-8'))
if __name__=='__main__': unittest.main()
