Three steps: 
1. ciq data processing
	基本照抄参考代码
2. pitchbook data processing
	把stata文件和Person.csv concat
	根据companyid补全'CompanyAlsoKnownAs', 'CompanyLegalName', 'CompanyFormerName'
	最后和'Startup_Investor_profile.csv'merge
3. Matching
	首先根据名字匹配df，用match_strings（最慢的地方）
	然后是用company name筛选删减一大部分的df
	接着根据'PrimaryInvestorType'分为nonangel和angel
	对angel的bio进行筛选
	最后把nonangel和angel合并
**为了方便，我将pitchbook里的'personid'改成了'LeadPartnerID'
**columns name相当麻烦，我遇到的大部分bug来自columns name没对上

match的结果：/nas_01/private/luguangli03/match pitchbook people with ciq people/match/pitchbook_capitaliq_final_match.pkl
没match上的：/nas_01/private/luguangli03/match pitchbook people with ciq people/match/Person_not_match.pkl