import bisect
import math
import unittest
from conditional_forecast import (ConditionalHawkes, Segment, validate_schedule,
    direct_history_intensity, advance)


class ForecastTests(unittest.TestCase):
    def close(self,x,y,tol=2e-10):
        self.assertAlmostEqual(x,y,delta=tol*max(1,abs(y)))

    def test_one_kernel_closed_form(self):
        a,b,mu,q0=.4,2.3,3.0,.7
        model=ConditionalHawkes([a],[b]); qe=a*mu/(b*(1-a))
        for h in [1e-9,.001,.3,10,600]:
            q,n=model.propagate([q0],h,mu)
            decay=b*(1-a)
            self.close(q[0],qe+(q0-qe)*math.exp(-decay*h))
            self.close(n,mu*h/(1-a)+b*(q0-qe)*(-math.expm1(-decay*h))/decay)

    def test_poisson_and_piecewise_baseline(self):
        model=ConditionalHawkes([0,0],[5,678])
        s=validate_schedule([Segment(0,1,2),Segment(1,3,5)])
        result=model.forecast([0,0],.2,.5,2,s)
        self.close(result['conditional_intensity_at_start'],2)
        self.close(result['expected_events_in_interval'],6)

    def test_equilibrium_long_and_short_horizons(self):
        model=ConditionalHawkes([.5312905395040888,.1875214503271147],[678.5018563404436,17.225593744579786])
        mu=1.2
        q=[a*mu/(b*(1-model.n)) for a,b in zip(model.alpha,model.beta)]
        for h in [1e-9,.001,.1,1,600,86400]:
            end,count=model.propagate(q,h,mu)
            for x,y in zip(end,q):self.close(x,y,3e-9)
            self.close(count,mu*h/(1-model.n),3e-9)

    def test_repeated_and_nearly_repeated_decay(self):
        single=ConditionalHawkes([.6],[5])
        for b2 in [5,5+1e-12]:
            double=ConditionalHawkes([.2,.4],[5,b2])
            for h in [.001,1,600]:
                q,n=double.propagate([.1,.2],h,2)
                q1,n1=single.propagate([.3],h,2)
                self.close(sum(q),q1[0]);self.close(n,n1)

    def test_near_critical_long_horizon(self):
        model=ConditionalHawkes([.9999],[4])
        h,mu,q0=10000,2,0
        q,n=model.propagate([q0],h,mu)
        decay=4*(1-.9999);qe=.9999*mu/decay
        self.close(q[0],qe*(-math.expm1(-decay*h)),1e-8)
        self.close(n,mu*h/(1-.9999)-4*qe*(-math.expm1(-decay*h))/decay,1e-8)

    def test_semigroup_and_integrated_count(self):
        m=ConditionalHawkes([.2,.4],[2,17]);q=[.8,.2]
        whole,n=m.propagate(q,1.9,3)
        mid,n1=m.propagate(q,.7,3);end,n2=m.propagate(mid,1.2,3)
        for x,y in zip(whole,end):self.close(x,y)
        self.close(n,n1+n2)

    def test_subinterval_and_baseline_boundary(self):
        m=ConditionalHawkes([.3,.2],[3,20]);q=[.1,.2]
        s=validate_schedule([Segment(0,.5,2),Segment(.5,1.7,8),Segment(1.7,3,1)])
        f=m.forecast(q,.2,.9,2.8,s)
        _,total=advance(m,q,.2,2.8,s);_,before=advance(m,q,.2,.9,s)
        self.close(f['expected_events_in_interval'],total-before)
        qa,_=m.propagate(q,.3,2);qb,_=m.propagate(qa,.4,8)
        self.close(f['conditional_intensity_at_start'],8+sum(b*x for b,x in zip(m.beta,qb)))

    def test_history_information_cutoff(self):
        m=ConditionalHawkes([.2,.3],[2,20]);events=[0,.1,.1,.5,2]
        cuts=[0,.1,.2,.8,2]
        states=list(m.history_states(events,cuts))
        for cutoff,state in zip(cuts,states):
            direct=m.state_from_history(events[:bisect.bisect_right(events,cutoff)],cutoff)
            for x,y in zip(state,direct):self.close(x,y)
        changed=list(m.history_states([0,.1,.1,.5,500],cuts[:-1]))
        self.assertEqual(states[:-1],changed)
        with self.assertRaises(ValueError):m.state_from_history(events,.8)

    def test_zero_future_immigration_still_allows_descendants(self):
        m=ConditionalHawkes([.5],[2]);s=[Segment(0,10,0)]
        exact=m.forecast([.7],0,1,2,s)
        old=direct_history_intensity(m,[.7],0,1,s)
        self.assertGreater(exact['conditional_intensity_at_start'],old)
        self.close(exact['conditional_intensity_at_start'],1.4*math.exp(-1))

    def test_independent_rk4_state_and_count(self):
        m=ConditionalHawkes([.2,.4],[2,17]);mu=3;q0=[.1,.3];h=1.3
        # Independent direct integration of [q1,q2, accumulated count].
        y=q0+[0.0];steps=5000;dt=h/steps
        def derivative(x):
            lam=mu+sum(b*z for b,z in zip(m.beta,x[:2]))
            return [-b*z+a*lam for a,b,z in zip(m.alpha,m.beta,x[:2])]+[lam]
        for _ in range(steps):
            k1=derivative(y);k2=derivative([x+dt*k/2 for x,k in zip(y,k1)])
            k3=derivative([x+dt*k/2 for x,k in zip(y,k2)]);k4=derivative([x+dt*k for x,k in zip(y,k3)])
            y=[x+dt*(u+2*v+2*w+z)/6 for x,u,v,w,z in zip(y,k1,k2,k3,k4)]
        q,n=m.propagate(q0,h,mu)
        for x,z in zip([*q,n],y):self.close(x,z,1e-9)

    def test_invalid_inputs(self):
        for a,b in [([1],[2]),([-.1],[2]),([.1],[0]),([float('nan')],[2])]:
            with self.assertRaises(ValueError):ConditionalHawkes(a,b)
        with self.assertRaises(ValueError):validate_schedule([Segment(0,1,2),Segment(2,3,2)])
        m=ConditionalHawkes([.2],[2])
        with self.assertRaises(ValueError):m.propagate([.1],-1e-13,2)
        with self.assertRaises(ValueError):m.forecast([.1],1,.5,2,[Segment(0,3,2)])
        with self.assertRaises(ValueError):m.forecast([],1.,1.,1.,[Segment(0,3,2)])
        with self.assertRaises(ValueError):m.state_from_history([],1.,float("nan"))

if __name__=='__main__':unittest.main(verbosity=2)
