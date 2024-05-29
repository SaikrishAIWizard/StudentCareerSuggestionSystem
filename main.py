from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import base64
from io import BytesIO

global data, predictions,predict_class

global career_paths_science_stream, career_paths_art_stream, career_paths_commerce_stream, career_paths_humanities_stream, career_paths_undecided_stream


career_paths_science_stream = """
Here are the top 5 career paths in the science stream after completing 10th class, with a brief roadmap:

1. Research Scientist:
   - Complete 12th grade with science.
   - Pursue a Bachelor's degree in a science field.
   - Obtain a Master's and/or Ph.D. for advanced roles.

2. Medical Professional:
   - Complete 12th grade with science.
   - Pursue MBBS/BDS/BAMS/BHMS.
   - Complete internship and acquire a license.

3. Data Scientist:
   - Complete 12th grade with science and math.
   - Obtain a Bachelor's in computer science or related.
   - Gain experience in programming and data analysis.

4. Biotechnologist:
   - Complete 12th grade with science.
   - Pursue a Bachelor's in Biotechnology.
   - Obtain a Master's for advanced roles.

5. Environmental Scientist:
   - Complete 12th grade with science.
   - Pursue a Bachelor's in Environmental Science.
   - Gain experience through internships and fieldwork.
"""

career_paths_art_stream = """
Here are the top 5 career paths in the arts stream after completing 10th class, with a brief roadmap:

1. Lawyer:
   - Complete 12th grade with arts.
   - Pursue a Bachelor's in Law (LLB).
   - Complete an internship and pass the bar exam.

2. Journalist:
   - Complete 12th grade with arts.
   - Pursue a Bachelor's in Journalism or Mass Communication.
   - Gain experience through internships and writing opportunities.

3. Psychologist:
   - Complete 12th grade with arts.
   - Pursue a Bachelor's in Psychology.
   - Obtain a Master's and/or Ph.D. for advanced roles.

4. Social Worker:
   - Complete 12th grade with arts.
   - Pursue a Bachelor's in Social Work.
   - Gain experience through internships and fieldwork.

5. Graphic Designer:
   - Complete 12th grade with arts.
   - Pursue a Bachelor's in Graphic Design or Visual Arts.
   - Build a strong portfolio and gain experience through internships or freelance work.
"""

career_paths_commerce_stream = """
Here are the top 5 career paths in the commerce stream after completing 10th class, with a brief roadmap:

1. Chartered Accountant:
   - Complete 12th grade with commerce.
   - Enroll for CA Foundation.
   - Complete CA Intermediate and Final examinations.

2. Financial Analyst:
   - Complete 12th grade with commerce and math.
   - Pursue a Bachelor's in Finance or Economics.
   - Gain experience through internships and certifications.

3. Company Secretary:
   - Complete 12th grade with commerce.
   - Enroll for CS Foundation.
   - Complete CS Executive and Professional examinations.

4. Economist:
   - Complete 12th grade with commerce and math.
   - Pursue a Bachelor's in Economics.
   - Obtain a Master's and/or Ph.D. for advanced roles.

5. Business Manager:
   - Complete 12th grade with commerce.
   - Pursue a Bachelor's in Business Administration (BBA).
   - Gain experience through internships and entry-level positions.
"""

career_paths_humanities_stream = """
Here are the top 5 career paths in the humanities stream after completing 10th class, with a brief roadmap:

1. Lawyer:
   - Complete 12th grade with humanities.
   - Pursue a Bachelor's in Law (LLB).
   - Complete an internship and pass the bar exam.

2. Psychologist:
   - Complete 12th grade with humanities.
   - Pursue a Bachelor's in Psychology.
   - Obtain a Master's and/or Ph.D. for advanced roles.

3. Social Worker:
   - Complete 12th grade with humanities.
   - Pursue a Bachelor's in Social Work.
   - Gain experience through internships and fieldwork.

4. Historian:
   - Complete 12th grade with humanities.
   - Pursue a Bachelor's in History.
   - Obtain a Master's and/or Ph.D. for advanced roles.

5. Journalist:
   - Complete 12th grade with humanities.
   - Pursue a Bachelor's in Journalism or Mass Communication.
   - Gain experience through internships and writing opportunities.
"""

sports_career_paths = """
Here are the top 5 career paths in the sports stream after completing 10th class, with a brief roadmap:

1. Professional Athlete:
   - Specialize in a particular sport.
   - Join a local sports club or academy.
   - Train rigorously and participate in local, national, and international competitions.
   - Secure sponsorships and endorsements.

2. Sports Coach:
   - Choose a sport or specialization.
   - Pursue coaching certifications and courses.
   - Gain coaching experience by assisting experienced coaches or mentoring younger athletes.
   - Continuously upgrade skills and knowledge.

3. Sports Physical Therapist:
   - Complete 12th grade with science.
   - Pursue a Bachelor's in Physical Therapy or Sports Medicine.
   - Obtain relevant certifications and licenses.
   - Gain experience through internships and working with sports teams.

4. Sports Nutritionist:
   - Complete 12th grade with science.
   - Pursue a Bachelor's in Nutrition or Dietetics.
   - Specialize in sports nutrition through additional certifications or courses.
   - Work with athletes to optimize their nutrition for performance.

5. Sports Journalist/Reporter:
   - Complete 12th grade with any stream.
   - Pursue a Bachelor's in Journalism or Mass Communication.
   - Gain experience through internships or freelancing for sports publications.
   - Develop strong writing and reporting skills in sports journalism.
"""

stock_market_career_path = """
After completing 12th grade, pursue a career in the stock market with the following steps:

1. Education: Obtain a Bachelor's degree in Finance, Economics, Business Administration, or related fields. Focus on courses covering financial markets, investment analysis, and financial modeling.

2. Gain Knowledge: Engage in self-study and research to understand market dynamics, trading strategies, technical analysis, and fundamental analysis. Stay updated with market news and trends through financial publications, online resources, and seminars.

3. Internships: Seek internships with brokerage firms, investment banks, or financial institutions to gain practical experience. Learn about market operations, trading platforms, and client interactions.

4. Certifications: Obtain relevant certifications such as Chartered Financial Analyst (CFA), Financial Risk Manager (FRM), or Series 7 license. These certifications demonstrate expertise and credibility in the field.

5. Start Trading: Open a brokerage account and start trading stocks. Begin with paper trading to practice strategies and risk management techniques without real money. Gradually transition to live trading with a well-defined trading plan and risk management strategy.

6. Continuous Learning: Stay updated with market trends, regulatory changes, and new technologies. Attend workshops, webinars, and conferences to enhance skills and network with industry professionals.

7. Specialization: Consider specializing in a specific area such as equity research, portfolio management, technical analysis, or algorithmic trading. Gain expertise in your chosen field through experience and further education.

By following this roadmap and continuously improving your skills and knowledge, you can build a successful career in the dynamic and rewarding field of the stock market.
"""

accountant_career_path = """
After completing 12th grade, pursue a career as an accountant with the following detailed roadmap:

1. Education: Obtain a Bachelor's degree in Accounting, Finance, or Business Administration from a reputable institution. Focus on courses covering financial accounting, managerial accounting, taxation, auditing, and financial reporting standards.

2. Certifications: Pursue professional certifications such as Certified Public Accountant (CPA), Chartered Accountant (CA), Certified Management Accountant (CMA), or Association of Chartered Certified Accountants (ACCA). These certifications enhance credibility and open doors to advanced career opportunities.

3. Internships: Gain practical experience through internships with accounting firms, corporations, or government agencies. Learn about accounting software, financial analysis, and regulatory compliance. Internships provide valuable insights into real-world accounting practices.

4. Entry-Level Position: Start your career as a staff accountant, junior auditor, or accounting clerk. Focus on building foundational skills in bookkeeping, financial analysis, and tax preparation. Learn from experienced professionals and take on increasing responsibilities.

5. Continuing Education: Stay updated with changes in accounting standards, tax laws, and technology. Pursue continuing education through workshops, seminars, and online courses. Consider pursuing a Master's degree or specialized certifications to advance your career.

6. Specialization: Explore specialized areas of accounting such as forensic accounting, tax accounting, internal auditing, or financial planning. Develop expertise in your chosen field through experience and additional training.

7. Career Advancement: Progress to roles such as senior accountant, controller, finance manager, or partner in an accounting firm. Focus on developing leadership skills, managing teams, and providing strategic financial guidance to organizations.

By following this roadmap and continuously enhancing your skills and knowledge, you can build a successful career as an accountant and make significant contributions to the financial success of businesses and organizations.
"""

banker_career_path = """
After completing 12th grade, aspiring bankers can follow this detailed roadmap:

1. Education: Pursue a Bachelor's degree in Finance, Economics, Business Administration, or a related field from a reputable institution. Focus on courses covering banking principles, financial markets, risk management, and accounting.

2. Internships: Gain practical experience through internships with banks, financial institutions, or investment firms. Internships provide exposure to banking operations, customer service, financial analysis, and regulatory compliance.

3. Entry-Level Position: Start your career as a bank teller, customer service representative, or financial analyst. Focus on building strong communication skills, attention to detail, and understanding of banking products and services.

4. Certifications: Obtain relevant certifications such as Certified Financial Planner (CFP), Chartered Financial Analyst (CFA), or Certified Banking Professional (CBP). These certifications demonstrate expertise and enhance career prospects.

5. Networking: Build professional connections within the banking industry through networking events, industry conferences, and online platforms. Networking can lead to job opportunities, mentorship, and valuable insights into the industry.

6. Specialization: Explore specialized areas of banking such as retail banking, commercial banking, investment banking, or wealth management. Develop expertise in your chosen area through experience, training, and continuous learning.

7. Career Advancement: Progress to roles such as branch manager, loan officer, investment banker, or financial advisor. Focus on developing leadership skills, managing client relationships, and achieving sales targets.

By following this roadmap and continuously improving your skills and knowledge, you can build a successful career in the dynamic and rewarding field of banking.
"""

scientist_career_path = """
After completing 12th grade, aspiring scientists can follow this detailed roadmap:

1. Education: Pursue a Bachelor's degree in a scientific field such as Physics, Chemistry, Biology, Environmental Science, or Engineering from a reputable institution. Focus on courses covering fundamental principles, laboratory techniques, and research methods.

2. Research Experience: Gain hands-on research experience through internships, summer programs, or research assistant positions in laboratories or academic institutions. Participate in research projects, conduct experiments, and analyze data under the guidance of experienced scientists.

3. Advanced Degrees: Consider pursuing a Master's degree or Ph.D. in your field of interest to deepen your knowledge and expertise. Graduate studies provide opportunities for independent research, specialization, and collaboration with leading researchers in the field.

4. Publish Research: Contribute to scientific knowledge by publishing research papers in peer-reviewed journals, presenting findings at conferences, and participating in scientific discussions. Publishing research enhances your visibility in the scientific community and establishes credibility as a scientist.

5. Collaborations and Networking: Build collaborations with researchers, professors, and professionals in your field through networking events, seminars, and scientific conferences. Collaborations foster interdisciplinary research, exchange of ideas, and access to resources and funding opportunities.

6. Grants and Funding: Apply for grants, fellowships, and research funding from government agencies, private foundations, and academic institutions to support your research projects. Securing funding allows you to pursue innovative research ideas and advance scientific knowledge.

7. Career Pathways: Explore career opportunities in academia, government research agencies, industry, non-profit organizations, or entrepreneurship. Choose a career path that aligns with your interests, skills, and long-term goals as a scientist.

By following this roadmap and continuously engaging in research, learning, and collaboration, you can build a successful career as a scientist and contribute to scientific discoveries and innovations.
"""

business_owner_career_path = """
After completing 12th grade, aspiring business owners can follow this detailed roadmap:

1. Identify Passion and Skills: Reflect on your interests, skills, and strengths to identify potential business ideas. Consider your passions, hobbies, and areas of expertise that can be translated into a viable business venture.

2. Market Research: Conduct market research to assess the demand for your product or service, identify target customers, understand competitors, and evaluate market trends. Market research helps validate your business idea and informs strategic decisions.

3. Business Plan: Develop a comprehensive business plan outlining your business concept, target market, marketing strategy, operational plan, financial projections, and growth objectives. A well-written business plan serves as a roadmap for your business and attracts investors and lenders.

4. Financing: Explore financing options to fund your business startup or expansion. Consider bootstrapping, personal savings, loans, venture capital, crowdfunding, or angel investors. Secure adequate funding to cover startup costs, initial inventory, equipment, and operating expenses.

5. Legal Structure and Registration: Choose a legal structure for your business such as sole proprietorship, partnership, limited liability company (LLC), or corporation. Register your business with the appropriate government authorities, obtain licenses and permits, and comply with regulatory requirements.

6. Branding and Marketing: Develop a strong brand identity including a memorable name, logo, and branding materials. Implement effective marketing strategies to promote your business, attract customers, and build brand awareness. Utilize digital marketing, social media, advertising, and networking to reach your target audience.

7. Launch and Operations: Launch your business and start operations according to your business plan. Establish efficient processes, hire skilled employees, manage finances, and provide excellent customer service. Adapt to market feedback, iterate on your business model, and continuously improve your products or services.

8. Growth and Expansion: Strategize for business growth and expansion by scaling operations, entering new markets, launching additional product lines, or diversifying revenue streams. Monitor key performance indicators (KPIs), track financial metrics, and adapt your business strategy to achieve long-term success.

By following this roadmap and leveraging your entrepreneurial skills, creativity, and determination, you can build a successful business and achieve your goals as a business owner.
"""

lawyer_career_path = """
After completing 12th grade, aspiring lawyers can follow this detailed roadmap:

1. Education: Pursue a Bachelor's degree in Law (LLB) from a reputable law school or university. Focus on courses covering legal principles, constitutional law, criminal law, civil procedure, contract law, and legal research and writing.

2. Internships and Clerkships: Gain practical experience through internships with law firms, government agencies, or legal aid organizations. Participate in clerkship programs to observe courtroom proceedings, assist attorneys, and develop legal skills.

3. Bar Exam Preparation: Prepare for and pass the bar exam in the jurisdiction where you intend to practice law. Study diligently, review legal concepts, practice sample questions, and take bar exam preparation courses to ensure success on the exam.

4. Legal Practice: Begin your legal career as an associate attorney at a law firm, public defender's office, prosecutor's office, or corporate legal department. Focus on building legal skills, researching case law, drafting legal documents, and representing clients in court.

5. Continuing Legal Education: Stay updated with changes in law, regulations, and legal precedents through continuing legal education (CLE) courses, seminars, and workshops. Maintain active licensure and fulfill CLE requirements to stay current in your practice area.

6. Specialization and Expertise: Explore specialized areas of law such as criminal law, family law, corporate law, environmental law, intellectual property law, or international law. Develop expertise in your chosen area through experience, training, and advanced legal education.

7. Professional Networking: Build professional relationships with fellow attorneys, judges, legal professionals, and clients through networking events, bar associations, legal conferences, and social media platforms. Networking enhances career opportunities, referrals, and mentorship.

8. Career Advancement: Progress to roles such as partner at a law firm, judge, government attorney, legal consultant, or corporate counsel. Focus on developing leadership skills, managing caseloads, and providing effective legal representation to clients.

By following this roadmap and demonstrating legal competency, ethical conduct, and commitment to justice, you can build a successful career as a lawyer and make a positive impact in the legal profession and society.
"""

doctor_career_path = """
After completing 12th grade, aspiring doctors can follow this detailed roadmap:

1. Education: Obtain a Bachelor's degree in a pre-medical field such as Biology, Chemistry, or Biochemistry from a reputable university. Focus on courses covering fundamental sciences, anatomy, physiology, and medical terminology.

2. Medical College Admission Test (MCAT): Prepare for and take the MCAT exam, which is required for admission to medical school. Study diligently, review science concepts, practice sample questions, and take MCAT preparation courses if necessary.

3. Medical School: Attend and complete four years of medical school to earn a Doctor of Medicine (MD) or Doctor of Osteopathic Medicine (DO) degree. Medical school curriculum includes classroom instruction, clinical rotations, and hands-on training in various medical specialties.

4. Residency Training: Complete a residency program in your chosen medical specialty to gain specialized training and clinical experience. Residency programs typically last three to seven years, depending on the specialty, and involve supervised patient care, surgeries, and research.

5. Board Certification: Obtain board certification in your medical specialty by passing the relevant board certification exam administered by the American Board of Medical Specialties (ABMS) or the American Osteopathic Association (AOA). Board certification demonstrates expertise and competence in your specialty.

6. Fellowship Training (Optional): Pursue additional fellowship training in a subspecialty within your medical specialty to further enhance your skills and knowledge. Fellowships provide advanced training and research opportunities in areas such as cardiology, oncology, or gastroenterology.

7. Licensure: Obtain a medical license from the state medical board in the state where you intend to practice medicine. Licensure requirements vary by state but typically include passing the United States Medical Licensing Examination (USMLE) or the Comprehensive Osteopathic Medical Licensing Examination (COMLEX-USA).

8. Continuing Medical Education (CME): Stay updated with advances in medicine, new treatments, and medical technologies through continuing medical education (CME) courses, seminars, and conferences. Maintain active licensure and fulfill CME requirements to stay current in your medical practice.

9. Professional Development: Engage in professional development activities such as research, publications, teaching, and leadership roles within medical organizations. Contribute to medical literature, mentor medical students, and participate in quality improvement initiatives to advance the field of medicine.

By following this roadmap and demonstrating dedication, compassion, and commitment to patient care, you can build a successful career as a doctor and make a positive impact on the health and well-being of individuals and communities.
"""

game_developer_career_path = """
After completing 12th grade, aspiring game developers can follow this detailed roadmap:

1. Education: Pursue a Bachelor's degree in Computer Science, Software Engineering, Game Development, or a related field from a reputable institution. Focus on courses covering programming languages, algorithms, data structures, and computer graphics.

2. Game Development Skills: Develop proficiency in game development tools, engines, and programming languages such as Unity, Unreal Engine, C++, C#, or Java. Build a portfolio of game projects showcasing your programming skills, creativity, and design abilities.

3. Specialization: Choose a specialization within game development such as game design, programming, graphics, animation, audio, or level design. Gain expertise in your chosen area through coursework, personal projects, and internships.

4. Internships and Projects: Gain practical experience through internships with game studios, indie developers, or software companies. Participate in game jams, hackathons, and collaborative projects to develop teamwork skills and expand your portfolio.

5. Networking: Build professional connections within the game development industry through networking events, game conferences, and online communities. Connect with game developers, artists, designers, and industry professionals to learn, collaborate, and share ideas.

6. Continuous Learning: Stay updated with advancements in game technology, industry trends, and best practices through online tutorials, workshops, and courses. Experiment with new tools, techniques, and game genres to expand your skill set and creativity.

7. Portfolio Development: Continuously update and improve your game development portfolio with new projects, demos, and prototypes. Showcase your best work, highlight your contributions, and demonstrate your ability to create engaging and innovative games.

8. Job Search and Career Growth: Apply for entry-level positions such as game programmer, game designer, or game artist at game studios, indie game companies, or software firms. Focus on building experience, mastering your craft, and advancing your career through promotions, additional training, and specialization.

By following this roadmap and continuously honing your skills, creativity, and passion for game development, you can build a successful career as a game developer and contribute to the exciting and dynamic world of video games.
"""

engineer_career_path = """
After completing 12th grade, aspiring engineers can follow this detailed roadmap:

1. Choose Engineering Discipline: Explore different engineering disciplines such as Mechanical Engineering, Electrical Engineering, Civil Engineering, Chemical Engineering, Computer Engineering, or Aerospace Engineering. Consider your interests, strengths, and career goals when selecting a discipline.

2. Education: Pursue a Bachelor's degree in your chosen engineering discipline from an accredited university or college. Focus on courses covering fundamental engineering principles, mathematics, physics, and specialized topics related to your field of study.

3. Hands-On Experience: Gain practical experience through internships, co-op programs, or research opportunities with engineering firms, government agencies, or academic institutions. Participate in engineering projects, design competitions, and laboratory experiments to apply theoretical knowledge and develop practical skills.

4. Professional Licensure: Obtain professional licensure or certification in your engineering discipline, if applicable. Many engineering fields require licensure to practice professionally and offer additional credentials such as Professional Engineer (PE) or Engineer-in-Training (EIT) designation.

5. Continuing Education: Stay updated with advancements in engineering technology, industry standards, and best practices through continuing education programs, professional development courses, and workshops. Pursue advanced degrees or specialized certifications to enhance your skills and career prospects.

6. Specialization: Choose a specialization within your engineering discipline such as robotics, renewable energy, biomedical engineering, structural design, or software engineering. Gain expertise in your chosen area through coursework, research projects, and hands-on experience.

7. Networking: Build professional relationships within the engineering community through networking events, industry conferences, and professional organizations such as the Institute of Electrical and Electronics Engineers (IEEE), American Society of Mechanical Engineers (ASME), or American Institute of Chemical Engineers (AIChE).

8. Career Development: Explore career opportunities in various sectors including manufacturing, construction, aerospace, automotive, energy, telecommunications, or technology. Consider pursuing advanced roles such as project manager, research scientist, engineering consultant, or executive leadership positions.

By following this roadmap and leveraging your technical skills, creativity, and problem-solving abilities, you can build a successful career as an engineer and contribute to innovation and progress in your chosen field.
"""

pharmacy_career_path = """
After completing 12th grade, aspiring pharmacists can follow this detailed roadmap:

1. Education: Pursue a Bachelor's degree in Pharmacy (B.Pharm) from a reputable pharmacy school or university. Focus on courses covering pharmaceutical sciences, pharmacology, medicinal chemistry, pharmacotherapy, and pharmacy practice.

2. Licensing Exam: Prepare for and pass the pharmacy licensing exam (e.g., NAPLEX in the United States) to become a licensed pharmacist. Study diligently, review pharmacy laws and regulations, and practice sample questions to ensure success on the exam.

3. Internships and Experiential Learning: Gain practical experience through internships, clerkships, or rotations in pharmacy settings such as community pharmacies, hospitals, clinics, or pharmaceutical companies. Learn about medication dispensing, patient counseling, drug therapy management, and pharmacy operations.

4. Residency Training (Optional): Consider completing a pharmacy residency program to gain advanced clinical training and specialization in areas such as ambulatory care, critical care, oncology, or psychiatric pharmacy. Residency programs typically last one to two years and involve direct patient care, research, and teaching.

5. Continuing Education: Stay updated with developments in pharmacy practice, drug therapy, and healthcare policies through continuing education programs, seminars, and conferences. Maintain active licensure and fulfill continuing education requirements to stay current in your practice area.

6. Specialization and Certification: Explore specialized areas of pharmacy practice such as geriatrics, pediatrics, infectious diseases, or ambulatory care. Obtain board certification in your chosen specialty (e.g., Board Certified Pharmacotherapy Specialist, Board Certified Ambulatory Care Pharmacist) to demonstrate expertise and enhance career opportunities.

7. Networking: Build professional connections within the pharmacy profession through networking events, pharmacy associations, and professional organizations such as the American Pharmacists Association (APhA) or the International Pharmaceutical Federation (FIP). Connect with pharmacists, healthcare providers, and industry professionals to learn, collaborate, and explore career opportunities.

8. Career Pathways: Explore career opportunities in various pharmacy settings including retail pharmacies, hospitals, long-term care facilities, pharmaceutical industry, academia, government agencies, or managed care organizations. Consider pursuing leadership roles, managerial positions, or entrepreneurial ventures in pharmacy practice.

By following this roadmap and demonstrating clinical expertise, professionalism, and dedication to patient care, you can build a successful career as a pharmacist and make a positive impact on healthcare delivery and patient outcomes.
"""

sports_career_path = """
After completing 12th grade, aspiring sportspeople can follow this detailed roadmap:

1. Identify Sporting Talent: Identify your sporting talents, interests, and strengths in specific sports or athletic disciplines. Assess your physical abilities, skills, and competitive spirit to determine your potential for success in sports.

2. Specialization: Choose a sport or athletic discipline to specialize in based on your interests, abilities, and long-term goals. Consider factors such as individual sports (e.g., tennis, swimming) or team sports (e.g., soccer, basketball) and indoor or outdoor sports.

3. Training and Development: Enroll in sports training programs, academies, or clubs to receive professional coaching and skill development. Participate in regular training sessions, practice drills, and physical conditioning to improve your performance, technique, and fitness level.

4. Competition and Performance: Compete in local, regional, and national-level sports competitions, tournaments, and championships to gain competitive experience and exposure. Set performance goals, challenge yourself against top athletes, and strive for personal bests in your chosen sport.

5. Education and Academic Balance: Maintain a balance between sports training and academic studies by prioritizing your education while pursuing sports excellence. Enroll in schools or colleges with sports-friendly programs, flexible schedules, and support for student-athletes.

6. Sports Nutrition and Fitness: Focus on sports nutrition, hydration, and physical fitness to optimize your athletic performance and recovery. Consult with sports nutritionists, strength and conditioning coaches, and sports medicine professionals to develop a personalized training regimen.

7. Sports Psychology and Mental Training: Develop mental toughness, focus, and resilience through sports psychology techniques, visualization, and mental training exercises. Learn strategies for managing stress, overcoming setbacks, and staying motivated in training and competition.

8. Injury Prevention and Rehabilitation: Prioritize injury prevention through proper warm-up, cool-down, and injury prevention exercises. Seek prompt medical attention for sports injuries and follow rehabilitation protocols to ensure a safe and speedy recovery.

9. Career Pathways: Explore career opportunities in sports such as professional athlete, coach, sports administrator, sports agent, sports commentator, or sports scientist. Pursue higher education, certifications, or specialized training to prepare for diverse roles in the sports industry.

By following this roadmap and demonstrating dedication, discipline, and resilience, you can pursue a successful career in sports and achieve your athletic goals.
"""

teacher_career_path = """
After completing 12th grade, aspiring teachers can follow this detailed roadmap:

1. Choose Teaching Subject: Identify your interests, strengths, and passion for teaching specific subjects or grade levels. Consider subjects such as English, Mathematics, Science, Social Studies, or languages, and determine whether you prefer teaching elementary, middle, or high school students.

2. Education: Pursue a Bachelor's degree in Education or a specific subject area from an accredited college or university. Complete coursework in teaching methods, curriculum development, educational psychology, and classroom management. Gain hands-on experience through practicum or student teaching placements.

3. Teacher Certification: Obtain teacher certification or licensure from the appropriate state or regional education board. Fulfill requirements such as passing teacher certification exams, completing a teaching internship, and meeting education and experience prerequisites.

4. Specialization and Endorsements: Consider pursuing specialized endorsements or certifications in areas such as special education, English as a Second Language (ESL), gifted education, or STEM education. Specializations enhance your qualifications and expand your teaching opportunities.

5. Classroom Experience: Gain classroom experience as a substitute teacher, teaching assistant, or volunteer tutor to build practical teaching skills and classroom management techniques. Observe experienced teachers, implement lesson plans, and interact with diverse student populations.

6. Professional Development: Participate in ongoing professional development opportunities such as workshops, conferences, and continuing education courses. Stay updated with educational trends, teaching strategies, and technology integration to enhance your teaching effectiveness.

7. Networking and Mentoring: Build professional relationships with fellow educators, mentors, and educational leaders through networking events, teacher associations, and online communities. Seek guidance from experienced teachers, collaborate on lesson planning, and share best practices.

8. Job Search and Career Growth: Apply for teaching positions at schools, districts, or educational organizations that align with your teaching philosophy and career goals. Pursue opportunities for career advancement such as lead teacher roles, department chair positions, or curriculum development roles.

9. Lifelong Learning and Reflection: Commit to lifelong learning and continuous improvement as a teacher by reflecting on your teaching practice, seeking feedback from colleagues and students, and implementing innovative teaching strategies. Stay inspired and passionate about making a difference in students' lives.

By following this roadmap and demonstrating dedication, empathy, and a commitment to student success, you can build a rewarding career as a teacher and positively impact the lives of generations of learners.
"""


entertainment_industry_career_path = """
After completing 12th grade, aspiring professionals in the entertainment industry can follow this detailed roadmap:

1. Self-Reflection and Skill Assessment: Reflect on your interests, talents, and career aspirations within the entertainment industry. Assess your skills in areas such as acting, filmmaking, music, dance, writing, or production to determine your niche.

2. Education and Training: Pursue formal education and training in your chosen field of entertainment. Enroll in acting schools, film academies, music conservatories, dance programs, or writing workshops to develop your craft and gain practical skills.

3. Skill Development and Portfolio Building: Hone your skills and build a portfolio of work to showcase your talent and creativity. Participate in student films, theater productions, music gigs, dance performances, or writing contests to gain experience and exposure.

4. Networking and Industry Connections: Build relationships with industry professionals, mentors, and peers through networking events, workshops, and online platforms. Attend industry conferences, film festivals, music showcases, or theater productions to connect with potential collaborators and mentors.

5. Internships and Entry-Level Opportunities: Seek internships, apprenticeships, or entry-level positions in entertainment companies, production studios, talent agencies, or media outlets. Gain hands-on experience, learn industry practices, and make valuable connections in the field.

6. Professional Development and Continuing Education: Stay updated with industry trends, techniques, and technologies through professional development opportunities. Take acting classes, filmmaking workshops, music lessons, or writing courses to enhance your skills and stay competitive.

7. Auditions and Casting Calls: Audition for acting roles, casting calls, or talent competitions to showcase your abilities and land opportunities in film, television, theater, or commercials. Prepare audition materials such as headshots, resumes, and demo reels to impress casting directors and producers.

8. Persistence and Resilience: Embrace rejection and setbacks as part of the entertainment industry journey. Stay resilient, persistent, and dedicated to your craft despite challenges and obstacles. Keep honing your skills, building relationships, and pursuing opportunities to achieve success in the industry.

9. Diversification and Adaptability: Explore diverse opportunities within the entertainment industry and adapt to changing trends and demands. Consider branching out into multiple disciplines or exploring new avenues such as digital media, streaming platforms, or independent projects.

By following this roadmap and leveraging your talent, passion, and perseverance, you can pursue a rewarding career in the dynamic and competitive entertainment industry.
"""




app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['GET', 'POST'])
def submit():
    global data, predictions,predict_class
    global career_paths_science_stream, career_paths_art_stream, career_paths_commerce_stream, career_paths_humanities_stream, career_paths_undecided_stream
    if request.method == 'POST':
        data = request.json
        print(data)  # Here you will have the data sent from the frontend
        # Your processing logic here
        # Load the model
        model = joblib.load('Model/ten_model.pkl')

        GENDER = data['gender']
        if GENDER.upper() == "MALE":
            GENDER = 1
        else:
            GENDER = 0
        HIN = int(data['HIN'])
        ENG = int(data['ENG'])
        MATH = int(data['MATH'])
        GEO = int(data['GEO'])
        HIST = int(data['HIST'])
        SCIENCE = int(data['SCIENCE'])
        BIO = int(data['BIO'])

        HOBBY = data["HOBBY"]
        if HOBBY == "Sports":
          HOBBY = 6
        elif HOBBY == "Gardening":
          HOBBY = 2
        elif HOBBY == "Music":
          HOBBY = 3
        elif HOBBY == "Cooking":
          HOBBY = 0
        elif HOBBY == "Photography":
          HOBBY = 4
        elif HOBBY == "Drawing":
          HOBBY = 1
        elif HOBBY == "Reading":
          HOBBY = 5
        elif HOBBY == "Travelling":
           HOBBY = 7


        INTEREST = data["INTEREST"]
        if INTEREST == "Music":
          INTEREST = 4
        elif INTEREST == "History":
          INTEREST = 3
        elif INTEREST == "Cooking":
          INTEREST = 1
        elif INTEREST == "Programming":
          INTEREST = 6
        elif INTEREST == "Art":
          INTEREST = 0
        elif INTEREST == "Dance":
         INTEREST = 2
        elif INTEREST == "Photography":
         INTEREST = 5
        elif INTEREST == "Science":
          INTEREST = 7

        SPORTS = data["SPORTS"]
        if SPORTS == "Swimming":
          SPORTS = 4
        elif SPORTS == "Table Tennis":
          SPORTS = 5
        elif SPORTS == "Tennis":
          SPORTS = 6
        elif SPORTS == "Volleyball":
          SPORTS = 7
        elif SPORTS == "Basketball":
          SPORTS = 1
        elif SPORTS == "Football":
          SPORTS = 3
        elif SPORTS == "Cricket":
         SPORTS = 2
        elif SPORTS == "Athletics":
          SPORTS = 0

        ACHIEVEMENTS = data["ACHIEVEMENTS"]
        if ACHIEVEMENTS == "Dance competition winner":
          ACHIEVEMENTS = 3
        elif ACHIEVEMENTS == "Chess competition winner":
          ACHIEVEMENTS = 2
        elif ACHIEVEMENTS == "Math Olympiad winner":
          ACHIEVEMENTS = 5
        elif ACHIEVEMENTS == "Music competition winner":
          ACHIEVEMENTS = 6
        elif ACHIEVEMENTS == "Debate competition winner":
          ACHIEVEMENTS = 4
        elif ACHIEVEMENTS == "Art competition winner":
          ACHIEVEMENTS = 0
        elif ACHIEVEMENTS == "Science fair winner ":
         ACHIEVEMENTS = 7
        elif ACHIEVEMENTS == "Athletics competition winner":
         ACHIEVEMENTS = 1

        LANGUAGE = data["LANGUAGE"]
        if LANGUAGE == "Hindi":
           LANGUAGE = 1
        elif LANGUAGE == "English":
            LANGUAGE = 0

        AVG = np.average([HIN,ENG,MATH,GEO,HIST,SCIENCE,BIO])
        new_data = np.array([GENDER,HIN,ENG,MATH,GEO,HIST,SCIENCE,BIO,HOBBY,INTEREST,SPORTS,ACHIEVEMENTS,LANGUAGE, AVG]).reshape(1,-1)

        predictions = model.predict(new_data)
        # Printing the predictions
        print(predictions)


        #Analysis Part
        # Predict probabilities
        probabilities = model.predict_proba(new_data)

        # Pie chart
        labels = model.classes_
        sizes = probabilities[0] *100  # Probabilities for the first sample, change index as needed

        # Encode classes along with probabilities
        encoded_classes = [f"{label}: {size:.2f}%" for label, size in zip(labels, sizes)]

        # Sort by sizes
        sorted_indices = np.argsort(sizes)[::-1]
        sorted_labels = [labels[i] for i in sorted_indices]
        sorted_sizes = sizes[sorted_indices]
        sorted_encoded_classes = [encoded_classes[i] for i in sorted_indices]
        # Create a DataFrame
        df_recomd = pd.DataFrame({"Class": sorted_labels, "Probability": sorted_sizes})


        # Plotting the data
        explode = [0.1 if p == max(sizes) else 0 for p in sizes]  # "explode" the largest slice
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))  # Increase the figure size

        fig.suptitle("Best stream suits for your career is: {}".format(predictions[0]), fontsize=12, fontweight='bold', y=0.96)


        # Pie Chart
        wedges, _, autotexts = ax1.pie(sorted_sizes, explode=explode, labels=sorted_encoded_classes, autopct='%1.1f%%', shadow=True, startangle=90)

        # Equal aspect ratio ensures that pie is drawn as a circle.
        ax1.axis('equal')

        # Adjust the layout
        plt.subplots_adjust(left=0.0, bottom=0.1, right=0.6)

        # Move the legend outside the pie chart
        ax1.legend(wedges, sorted_encoded_classes, title="Classes", loc="upper right", bbox_to_anchor=(1, 0.5, 1, 0.5))

        # Plot the table
        ax2.axis('tight')
        ax2.axis('off')
        table = ax2.table(cellText=df_recomd.values, colLabels=df_recomd.columns, loc='center')

        # Save the plot as an image
        plt.savefig('plot_image.png', bbox_inches='tight')  # Save the plot as an image

        return jsonify({"message": "Data received successfully!"})
    
    elif request.method == 'GET':
        # Open and encode the image
        with open("plot_image.png", "rb") as img_file:
            encoded_string = base64.b64encode(img_file.read()).decode('utf-8')

        if predictions[0] == "Science":
            img_path = "Images/science.jpg"
            roadmap = career_paths_science_stream
        elif predictions[0] == "Sports":
            img_path = "Images/undecided.jpg"
            roadmap = sports_career_paths
        elif predictions[0] == "Humanities":
            img_path = "Images/humanities.jpg"
            roadmap = career_paths_humanities_stream
        elif predictions[0] == "Commerce":
            img_path = "Images/commerce.jpg"
            roadmap = career_paths_commerce_stream
        elif predictions[0] == "Arts":
            img_path = "Images/arts.jpg"
            roadmap = career_paths_art_stream

        roadmap =  roadmap.replace("\n", "<br>")

        with open(img_path, "rb") as img_file:
            stream_img = base64.b64encode(img_file.read()).decode('utf-8')

        # Prepare the response
        predict_class = {
            "roadmap" : roadmap,
            "stream_img":stream_img,
            "image": encoded_string,
            "prediction": predictions[0]  # Example prediction
            }
        return jsonify(predict_class)
    

@app.route('/inter', methods=['GET', 'POST'])
def inter():
    global data, predictions,predict_class
    #global career_paths_science_stream, career_paths_art_stream, career_paths_commerce_stream, career_paths_humanities_stream
    global stock_market_career_path,accountant_career_path,banker_career_path,scientist_career_path,business_owner_career_path,lawyer_career_path,doctor_career_path,game_developer_career_path,engineer_career_path,pharmacy_career_path,sports_career_path,teacher_career_path,entertainment_industry_career_path
    if request.method == 'POST':
         data = request.json
         print(data)  # Here you will have the data sent from the frontend
         # Your processing logic here
         # Load the model
         model = joblib.load('Model/inter_model.pkl')

         academic_per = data['academic_per']
         hours_study = data['hours_study']
         Logical_quotient_rating = data['Logical_quotient_rating']
         hackathons = data['hackathons']
         coding_skills_rating = data['coding_skills_rating']
         public_speaking_points = data['public_speaking_points']

         self_learning = data['self_learning'] # yes or no
         if self_learning.lower() == "yes":
            self_learning = 1
         else:
            self_learning = 0

         Extra_courses = data['Extra_courses'] # yes or no
         if Extra_courses.lower() == "yes":
            Extra_courses = 1
         else:
            Extra_courses = 0


         certifications = data['certifications']  # Replace this with the desired certification value

         if certifications == "Pharmacy Technician Certification (CPhT)":
            certifications = 22
         elif certifications == "Diploma in Pharmaceutical Sciences":
            certifications = 18
         elif certifications == "Certified Accounting Technician (CAT)":
            certifications = 8
         elif certifications == "Investopedia: Stock Market Investing Course":
            certifications = 20
         elif certifications == "Diploma in Accounting":
            certifications = 17
         elif certifications == "Certificate in Financial Markets":
            certifications = 4
         elif certifications == "First Aid and CPR Certification":
            certifications = 19
         elif certifications == "Coursera: Introduction to Research Methods":
            certifications = 16
         elif certifications == "Certificate in Laboratory Techniques":
            certifications = 5
         elif certifications == "Coursera: Foundations of Business Strategy":
            certifications = 12
         elif certifications == "Certified Nursing Assistant (CNA)":
            certifications = 10
         elif certifications == "Certified Fitness Trainer (CFT)":
            certifications = 9
         elif certifications == "Coursera: Introduction to Game Development":
            certifications = 14
         elif certifications == "Certificate in Business Administration":
            certifications = 3
         elif certifications == "Coursera: Introduction to Law":
            certifications = 15
         elif certifications == "Unity Certified User":
            certifications = 25
         elif certifications == "Sports First Aid Certification":
            certifications = 23
         elif certifications == "Certificate in Legal Studies":
            certifications = 6
         elif certifications == "Certificate in Banking and Finance":
            certifications = 2
         elif certifications == "Coursera: Introduction to Digital Marketing":
            certifications = 13
         elif certifications == "Teaching Certificate/License":
            certifications = 24
         elif certifications == "National Board Certification (NBC)":
            certifications = 21
         elif certifications == "Associate Chartered Banker (ACIB)":
            certifications = 0
         elif certifications == "Certificate in Media Production":
            certifications = 7
         elif certifications == "AutoCAD Certification":
            certifications = 1
         elif certifications == "Certified SolidWorks Associate (CSWA)":
            certifications = 11
         else:
            certifications = -1  # Default value for unknown certification


         talenttests_taken = data['talenttests_taken'] # yes or no
         if talenttests_taken.lower() == "yes":
            talenttests_taken = 1
         else:
            talenttests_taken = 0

         olympiads = data['olympiads'] # yes or no
         if olympiads == "OfficeBased":
            olympiads = 1
         elif olympiads == "RemoteWork":
            olympiads = 2
         else:
            olympiads = 0



         reading_writing_skills = data["reading_writing_skills"]
         if reading_writing_skills.lower() == "excellent":
            reading_writing_skills = 0
         elif reading_writing_skills.lower() == "poor":
            reading_writing_skills = 2
         else:
            reading_writing_skills = 1

         memory_capability_score = data["memory_capability_score"]
         if memory_capability_score.lower() == "excellent":
            memory_capability_score = 0
         elif memory_capability_score.lower() == "poor":
            memory_capability_score = 2
         else:
            memory_capability_score = 1

         Interested_subjects = data["Interested_subjects"]  # Replace this with the desired subject

         if Interested_subjects == "Pharmacology":
            Interested_subjects = 18
         elif Interested_subjects == "Pharmaceutical Chemistry":
            Interested_subjects = 17
         elif Interested_subjects == "Financial Statement Analysis":
            Interested_subjects = 7
         elif Interested_subjects == "Financial Analytics and Data Visualization":
            Interested_subjects = 5
         elif Interested_subjects == "Forensic Accounting":
            Interested_subjects = 8
         elif Interested_subjects == "The Intelligent Investor":
            Interested_subjects = 22
         elif Interested_subjects == "The Structure of Scientific Revolutions":
            Interested_subjects = 24
         elif Interested_subjects == "Human Anatomy and Physiology":
            Interested_subjects = 10
         elif Interested_subjects == "Medical Ethics and Professionalism":
            Interested_subjects = 16
         elif Interested_subjects == "Lab Girl":
            Interested_subjects = 11
         elif Interested_subjects == "Master the Art of Business":
            Interested_subjects = 13
         elif Interested_subjects == "The Lean Startup":
            Interested_subjects = 23
         elif Interested_subjects == "Sports Management and Administration":
            Interested_subjects = 20
         elif Interested_subjects == "Unity in Action":
            Interested_subjects = 25
         elif Interested_subjects == "Constitutional Law":
            Interested_subjects = 0
         elif Interested_subjects == "Sports Science and Exercise Physiology":
            Interested_subjects = 21
         elif Interested_subjects == "Game Programming Patterns":
            Interested_subjects = 9
         elif Interested_subjects == "Legal Research and Writing":
            Interested_subjects = 12
         elif Interested_subjects == "Corporate Finance":
            Interested_subjects = 1
         elif Interested_subjects == "Curriculum Development and Instructional Design":
            Interested_subjects = 2
         elif Interested_subjects == "Entertainment Business and Management":
            Interested_subjects = 4
         elif Interested_subjects == "Financial Markets":
            Interested_subjects = 6
         elif Interested_subjects == "Media Studies and Communication":
            Interested_subjects = 15
         elif Interested_subjects == "Educational Psychology":
            Interested_subjects = 3
         elif Interested_subjects == "Physics and Mechanics":
            Interested_subjects = 19
         elif Interested_subjects == "Mathematics and Applied Mathematics":
            Interested_subjects = 14
         else:
            Interested_subjects = -1  # Default value for unknown subject

         interested_career_area = "Management Consulting"  # Replace this with the desired career area

         if interested_career_area == "Clinical Pharmacy":
            interested_career_area = 2
         elif interested_career_area == "Pharmaceutical Research and Development":
            interested_career_area = 18
         elif interested_career_area == "International Taxation and Compliance":
            interested_career_area = 12
         elif interested_career_area == "Portfolio Management":
            interested_career_area = 19
         elif interested_career_area == "Financial Planning and Analysis":
            interested_career_area = 9
         elif interested_career_area == "Financial Analysis":
            interested_career_area = 8
         elif interested_career_area == "Public Health":
            interested_career_area = 22
         elif interested_career_area == "Research and Development":
            interested_career_area = 23
         elif interested_career_area == "Academic Research and Teaching":
            interested_career_area = 0
         elif interested_career_area == "Medical Research":
            interested_career_area = 16
         elif interested_career_area == "Entrepreneurship and Startup Management":
            interested_career_area = 6
         elif interested_career_area == "Game Design":
            interested_career_area = 10
         elif interested_career_area == "Management Consulting":
            interested_career_area = 15
         elif interested_career_area == "Sports Marketing and Sponsorship":
            interested_career_area = 24
         elif interested_career_area == "Sports Medicine and Athletic Training":
            interested_career_area = 25
         elif interested_career_area == "Corporate Law":
            interested_career_area = 3
         elif interested_career_area == "Game Programming":
            interested_career_area = 11
         elif interested_career_area == "Litigation":
            interested_career_area = 14
         elif interested_career_area == "Educational Technology Specialist":
            interested_career_area = 5
         elif interested_career_area == "Private Wealth Management":
            interested_career_area = 20
         elif interested_career_area == "Film and Television Production":
            interested_career_area = 7
         elif interested_career_area == "Investment Banking":
            interested_career_area = 13
         elif interested_career_area == "Music Industry Management":
            interested_career_area = 17
         elif interested_career_area == "Product Engineer":
            interested_career_area = 21
         elif interested_career_area == "Education Policy and Advocacy":
            interested_career_area = 4
         elif interested_career_area == "Artificial Intelligence and Machine Learning":
            interested_career_area = 1
         else:
            interested_career_area = -1  # Default value for unknown career area


         Job_Higher_Studies = data["Job_Higher_Studies"] 
         if Job_Higher_Studies.lower() == "higherstudies":
            Job_Higher_Studies = 1
         else:
            Job_Higher_Studies = 0


         Taken_inputs_seniors = data["Taken_inputs_seniors"]
         if Taken_inputs_seniors.lower() == "yes":
            Taken_inputs_seniors = 1
         else:
            Taken_inputs_seniors = 0

         interested_in_games = data["interested_in_games"]
         if interested_in_games.lower() == "yes":
            interested_in_games = 1
         else:
            interested_in_games = 0


         Interested_Books = data["Interested_Books"]
         if Interested_Books == "Guide":
            Interested_Books = 13
         elif Interested_Books == "Health":
            Interested_Books = 14
         elif Interested_Books == "Self help":
            Interested_Books = 27
         elif Interested_Books == "Horror":
            Interested_Books = 16
         elif Interested_Books == "Autobiographies":
            Interested_Books = 3
         elif Interested_Books == "Fantasy":
            Interested_Books = 12
         elif Interested_Books == "Satire":
            Interested_Books = 24
         elif Interested_Books == "Biographies":
            Interested_Books = 4
         elif Interested_Books == "Comics":
            Interested_Books = 6
         elif Interested_Books == "Poetry":
            Interested_Books = 20
         elif Interested_Books == "Encyclopedias":
            Interested_Books = 11
         elif Interested_Books == "Prayer books":
            Interested_Books = 21
         elif Interested_Books == "Anthology":
            Interested_Books = 1
         elif Interested_Books == "Science fiction":
            Interested_Books = 26
         elif Interested_Books == "Art":
            Interested_Books = 2
         elif Interested_Books == "History":
            Interested_Books = 15
         elif Interested_Books == "Mystery":
            Interested_Books = 19
         elif Interested_Books == "Diaries":
            Interested_Books = 8
         elif Interested_Books == "Drama":
            Interested_Books = 10
         elif Interested_Books == "Childrens":
            Interested_Books = 5
         elif Interested_Books == "Travel":
            Interested_Books = 29
         elif Interested_Books == "Religion-Spirituality":
            Interested_Books = 22
         elif Interested_Books == "Action and Adventure":
            Interested_Books = 0
         elif Interested_Books == "Trilogy":
            Interested_Books = 30
         elif Interested_Books == "Dictionaries":
            Interested_Books = 9
         elif Interested_Books == "Romance":
            Interested_Books = 23
         elif Interested_Books == "Science":
            Interested_Books = 25
         elif Interested_Books == "Series":
            Interested_Books = 28
         elif Interested_Books == "Cookbooks":
            Interested_Books = 7
         elif Interested_Books == "Journals":
            Interested_Books = 17
         else:
            Interested_Books = 18  #Math

         Salary_Range_Expected = data["Salary_Range_Expected"] # yes or no
         if Salary_Range_Expected.lower() == "salary":
            Salary_Range_Expected = 1
         else:
            Salary_Range_Expected = 0 # Work

         Gentle_Tuff = data["Gentle_Tuff"]
         if Gentle_Tuff.lower() == "gentle":
            Gentle_Tuff = 1
         else:
            Gentle_Tuff = 0 # stubborn

         Salary_work =  data["Salary_Range_Expected"]      #Salary_Range_Expected
         if Salary_work.lower() == "salary":
            Salary_work = 1
         else:
            Salary_work = 0 # Work

         hard_worker = data["hard_worker"]
         if hard_worker.lower() == "smart worker":
            hard_worker = 1
         else:
            hard_worker = 0 # hard worker

         Introvert = data["Introvert"]
         if Introvert.lower() == "yes":
            Introvert = 1
         else:
            Introvert = 0   
         
         new_data = np.array([academic_per,hours_study,Logical_quotient_rating,hackathons,coding_skills_rating,public_speaking_points,
 self_learning, Extra_courses,certifications, talenttests_taken, olympiads,reading_writing_skills,
 memory_capability_score,Interested_subjects,interested_career_area,Job_Higher_Studies,Taken_inputs_seniors,
 interested_in_games,Interested_Books,Salary_Range_Expected,Gentle_Tuff,Salary_work,hard_worker,Introvert]).reshape(1,-1)

         #test_data = df.iloc[-41].values[1:-1].reshape(1, -1)
         # Make predictions
         # Assuming you have new_data for which you want to make predictions
         predictions = model.predict(new_data)

         #Analysis Part
         # Predict probabilities
         probabilities = model.predict_proba(new_data)

         # Pie chart
         labels = model.classes_
         sizes = probabilities[0] *100  # Probabilities for the first sample, change index as needed

         # Encode classes along with probabilities
         encoded_classes = [f"{label}: {size:.2f}%" for label, size in zip(labels, sizes)]

         # Sort by sizes
         sorted_indices = np.argsort(sizes)[::-1]
         sorted_labels = [labels[i] for i in sorted_indices]
         sorted_sizes = sizes[sorted_indices]
         sorted_encoded_classes = [encoded_classes[i] for i in sorted_indices]
         # Create a DataFrame
         df_recomd = pd.DataFrame({"Class": sorted_labels, "Probability": sorted_sizes})


         # Filter labels with less than 5% probability
         threshold = 0.5
         filtered_indices = np.where(sorted_sizes >= threshold)
         filtered_labels = [sorted_labels[i] for i in filtered_indices[0]]
         filtered_sizes = sorted_sizes[filtered_indices]
         filtered_encoded_classes = [sorted_encoded_classes[i] for i in filtered_indices[0]]

         # Plotting the data
         explode = [0.1 if p == max(filtered_sizes) else 0 for p in filtered_sizes]  # "explode" the largest slice
         fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))  # Increase the figure size
         fig.suptitle("Best stream suits for your career is: {}".format(predictions[0]), fontsize=12, fontweight='bold', y=0.96)

         # Pie Chart
         wedges, _, autotexts = ax1.pie(filtered_sizes, explode=explode, labels=filtered_encoded_classes, autopct='%1.1f%%', shadow=True, startangle=180)

         # Equal aspect ratio ensures that pie is drawn as a circle.
         ax1.axis('equal')

         # Adjust the layout
         plt.subplots_adjust(left=0.0, bottom=0.1, right=0.6)

         # Move the legend outside the pie chart
         ax1.legend(wedges, filtered_encoded_classes, title="Classes", loc="lower center", bbox_to_anchor=(1, 0, 1, 0))

         # Plot the table
         ax2.axis('tight')
         ax2.axis('off')
         table = ax2.table(cellText=df_recomd.values, colLabels=df_recomd.columns,loc="best")
         # Save the plot as an image
         plt.savefig('plot_image.png', bbox_inches='tight')  # Save the plot as an image

         return jsonify({"message": "Data received successfully!"})
    
    elif request.method == 'GET':
        # Open and encode the image
        with open("plot_image.png", "rb") as img_file:
            encoded_string = base64.b64encode(img_file.read()).decode('utf-8')
        
        if predictions[0] == "Science":
            img_path = "Images/science.jpg"
            roadmap = career_paths_science_stream
        elif predictions[0] == "Sports":
            img_path = "Images/undecided.jpg"
            roadmap = sports_career_paths
        elif predictions[0] == "Humanities":
            img_path = "Images/humanities.jpg"
            roadmap = career_paths_humanities_stream
        elif predictions[0] == "Commerce":
            img_path = "Images/commerce.jpg"
            roadmap = career_paths_commerce_stream
        elif predictions[0] == "Arts":
            img_path = "Images/arts.jpg"
            roadmap = career_paths_art_stream
        elif predictions[0] == "Stock Investor":
            img_path = "Images/Stock_Investor.jpg"
            roadmap = stock_market_career_path
        elif predictions[0] == "Accountant":
            img_path = "Images/Accountant.jpg"
            roadmap = accountant_career_path
        elif predictions[0] == "Banker":
            img_path = "Images/Banker.jpg"
            roadmap = banker_career_path
        elif predictions[0] == "Scientist":
            img_path = "Images/Scientist.jpg"
            roadmap = scientist_career_path
        elif predictions[0] == "MBA":
            img_path = "Images/Business_Owner.jpg"
            roadmap = business_owner_career_path
        elif predictions[0] == "Lawyer":
            img_path = "Images/Lawyer.jpg"
            roadmap = lawyer_career_path
        elif predictions[0] == "Doctor":
            img_path = "Images/Doctor.jpg"
            roadmap = doctor_career_path
        elif predictions[0] == "Game Developer":
            img_path = "Images/Game_Developer.jpg"
            roadmap = game_developer_career_path
        elif predictions[0] == "Engineer":
            img_path = "Images/Software_Engineer.jpg"
            roadmap = engineer_career_path
        elif predictions[0] == "Pharmacy":
            img_path = "Images/Software_Engineer.jpg"
            roadmap = pharmacy_career_path
        elif predictions[0] == "Sports person":
            img_path = "Images/Software_Engineer.jpg"
            roadmap = sports_career_path
        elif predictions[0] == "Teacher":
            img_path = "Images/Software_Engineer.jpg"
            roadmap = teacher_career_path 
        elif predictions[0] == "Entertainment industry":
            img_path = "Images/Software_Engineer.jpg"
            roadmap = entertainment_industry_career_path    
        roadmap =  roadmap.replace("\n", "<br>")
           
        with open(img_path, "rb") as img_file:
            stream_img = base64.b64encode(img_file.read()).decode('utf-8')

        # Prepare the response
        predict_class = {
            "roadmap" : roadmap,
            "stream_img":stream_img,
            "image": encoded_string,
            "prediction": predictions[0]  # Example prediction
            }
        return jsonify(predict_class)


if __name__ == '__main__':
    app.run(debug=True)
