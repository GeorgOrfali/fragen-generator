import xml.etree.ElementTree as ET
import json
import random

class JACKXML():
    def generate(self, aufgaben):
        root = ET.Element('Exercise', id="1")
        ET.SubElement(root, 'name').text = 'Generierte Aufgabe'
        ET.SubElement(root, 'publicDescription')
        ET.SubElement(root, 'internalNotes')
        ET.SubElement(root, 'language').text = 'de'
        ET.SubElement(root, 'difficulty').text = '0'
        ET.SubElement(root, 'tags', id="2")
        ET.SubElement(root, 'resources', id="3")
        stages = ET.SubElement(root, 'stages', id="4")
        aufgaben = json.loads(aufgaben)
        aid = 5
        for aufgabe in aufgaben:
            if aufgabe is not None:
                antworten = []
                if aufgabe['type'] == 'Wahr-Falsch' or aufgabe['type'] == 'Single Choice':
                    mcstage = ET.SubElement(stages, 'MCStage', id=str(aid))
                    ET.SubElement(mcstage, 'internalName').text = '#' + str(aid)
                    ET.SubElement(mcstage, 'taskDescription').text = aufgabe['question']
                    if aufgabe['type'] == 'Wahr-Falsch':
                        antworten.append({'answer': aufgabe['answer'], 'rule': 'CORRECT'})
                        if aufgabe['answer'] == ' Wahr':
                            antworten.append({'answer': 'Falsch', 'rule': 'WRONG'})
                        else:
                            antworten.append({'answer': 'Wahr', 'rule': 'WRONG'})
                    else:
                        antworten.append({'answer': aufgabe['answer'], 'rule': 'CORRECT'})
                        distractors = aufgabe['distractors'].split(',')

                        for d in distractors:
                            antworten.append({'answer': d, 'rule': 'WRONG'})

                    random.shuffle(antworten)
                    ET.SubElement(mcstage, 'allowSkip').text = 'false'
                    answers = ET.SubElement(mcstage, 'answerOptions')
                    for antwort in antworten:
                        mcanswers = ET.SubElement(answers, 'MCAnswer')
                        ET.SubElement(mcanswers, 'rule').text = antwort['rule']
                        ET.SubElement(mcanswers, 'text').text = antwort['answer']
                        ET.SubElement(mcanswers, 'mcstage', reference=str(aid))
                    ET.SubElement(mcstage, 'singleChoice').text = 'true'

                elif aufgabe['type'] == 'LückenText':
                    fillstage = ET.SubElement(stages, 'FillInStage', id=str(aid))
                    ET.SubElement(fillstage, 'internalName').text = '#' + str(aid)
                    question = aufgabe['question'].replace('/blank', '&amp;nbsp;&lt;input name=&quot;fillin'+str(aid)+'&quot; size=&quot;10&quot; type=&quot;text&quot; value=&quot;fillin1&quot; /&gt;')
                    ET.SubElement(fillstage, 'taskDescription').text = question
                    ca = ET.SubElement(fillstage, 'correctAnswerRules')
                    r = ET.SubElement(ca, 'Rule')
                    ve = ET.SubElement(r, 'validationExpression')
                    ET.SubElement(ve,'code').text = 'equals([input=fillin'+str(aid)+'], &apos;'+aufgabe['answer']+'&apos;)'


                aid = aid + 1
        self.indent(root)
        tree = ET.ElementTree(root)
        tree.write('generierte_uebung.xml')

    def indent(self, elem, level=0):
        i = "\n" + level*"  "
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = i + "  "
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
            for elem in elem:
                self.indent(elem, level+1)
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = i