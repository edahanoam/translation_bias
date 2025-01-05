from xml.etree.ElementTree import Element, SubElement, ElementTree


def generate_tmx(source_file, target_file, srclang, tgtlang, output_file):
    # Create the TMX root and header
    root = Element('tmx', version="1.4")
    header = SubElement(root, 'header', {
        'creationtool': 'CustomScript',
        'creationtoolversion': '1.0',
        'segtype': 'sentence',
        'adminlang': 'en-US',
        'srclang': srclang,
        'datatype': 'plaintext'
    })
    body = SubElement(root, 'body')

    # Read the source and target files
    with open(source_file, 'r', encoding='utf-8') as src, open(target_file, 'r', encoding='utf-8') as tgt:
        for source_line, target_line in zip(src, tgt):
            source_line = source_line.strip()
            target_line = target_line.strip()

            # Skip empty lines
            if not source_line or not target_line:
                continue

            # Create translation units
            tu = SubElement(body, 'tu')
            tuv_src = SubElement(tu, 'tuv', {'xml:lang': srclang})
            seg_src = SubElement(tuv_src, 'seg')
            seg_src.text = source_line
            tuv_tgt = SubElement(tu, 'tuv', {'xml:lang': tgtlang})
            seg_tgt = SubElement(tuv_tgt, 'seg')
            seg_tgt.text = target_line

    # Write the TMX file
    tree = ElementTree(root)
    with open(output_file, 'wb') as file:
        tree.write(file, encoding='UTF-8', xml_declaration=True)


# Example usage
source_file = 'pilot_pro_sentences.txt'  # Replace with your source file path
target_file = 'translation_check_pro.txt'  # Replace with your target file path
srclang = 'en'  # English as the source language
tgtlang = 'he'  # Hebrew as the target language
output_file = 'translations.tmx'

generate_tmx(source_file, target_file, srclang, tgtlang, output_file)
