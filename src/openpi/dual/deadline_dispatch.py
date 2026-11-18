"""C commands per decision; hold or use an aligned old tail while inference is late."""
class Dispatch:
    def __init__(self,c,tail):
        self.c,self.tail=c,tail
        self.plan=None;self.adoption=None;self.last_gripper=-1.;self.offset=0
    def adopt(self,p,tick):
        lag=tick-p['decision_step'] if self.tail else 0
        if lag<0:raise ValueError('future decision')
        if lag>=len(p['actions']):return False
        self.plan=p;self.adoption=tick;self.offset=lag
        return True
    def need(self,next_tick):
        return next_tick-self.adoption>=self.c
    def command(self,tick):
        p=self.plan; age=tick-self.adoption
        slot=self.offset+age
        if (age>=self.c and not self.tail) or slot>=len(p['actions']):
            return [0.]*6+[self.last_gripper],'hold',None
        action=list(map(float,p['actions'][slot]));self.last_gripper=action[-1]
        src=('cold' if tick==0 else ('fresh' if tick==p['decision_step'] else 'late') if age==0 else 'committed' if age<self.c else 'tail')
        return action,src,slot
