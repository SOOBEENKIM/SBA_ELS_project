"""Settlement day arithmetic with an explicit, versioned Korean bank calendar."""
from pathlib import Path
from datetime import date,timedelta
import json

PATH=Path(__file__).resolve().parent/'evidence/kr_bank_calendar.json'

class SettlementCalendar:
    def __init__(self,path=PATH):
        self.spec=json.loads(Path(path).read_text());self.closed=set(self.spec['closed_days'])
    def business_day(self,d):
        if not self.spec['first_year']<=d.year<=self.spec['last_year']:raise ValueError('Calendar year outside supplied coverage')
        return d.weekday()<5 and d.isoformat() not in self.closed
    def advance(self,d,n):
        if isinstance(d,str):d=date.fromisoformat(d)
        if not isinstance(n,int) or n<0:raise ValueError('Nonnegative explicit business-day lag required')
        while n:
            d+=timedelta(days=1)
            if self.business_day(d):n-=1
        return d
