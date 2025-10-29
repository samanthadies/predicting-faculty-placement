"""
XMLToCSV.py

Parses the official DBLP XML dump using its DTD and writes one CSV per DBLP
element type (e.g., article, inproceedings, book). Optionally writes an
annotated header with inferred column types, generates Neo4j-friendly headers,
and emits relationship CSVs for selected attributes (e.g., authors).

Key features:
- Validates XML against the DTD (lxml).
- Discovers per-element attributes from the DTD-backed parse.
- Writes per-element CSVs with a stable ';' delimiter and 'id' column.
- Optionally annotates headers with inferred types and Neo4j-friendly schema.
- Optionally emits relationship CSVs for specified attributes (e.g., author).
- Can generate a `neo4j-admin import` command for convenience.

Usage:
python dblp_to_csv.py <dblp.xml> <dblp.dtd> <output.csv> [--annotate] [--neo4j]
                      [--relations author:authored_by year:has_year ...]

Notes:
- `--neo4j` implies `--annotate`.
- `--relations attr:REL` turns attribute `attr` into a node set with a
  relationship `REL` from each parent element row.


Source:
Adapted (commented and documented) from:
https://github.com/ThomHurks/dblp-to-csv

10/24/2025 --- SD
"""

import argparse
import csv
import os
import re
import time
from datetime import date, datetime
from typing import Dict, Tuple, Union

from lxml import etree

__author__ = 'Thom Hurks'


class InvalidElementName(Exception):
    """
    Raised when an invalid column/tag name is encountered while constructing CSV.

    Specifically used if an XML child tag produces a column named 'id', which
    collides with the synthetic row identifier we add.
    """
    def __init__(self, invalid_element_name, tag_name, parent_name):
        self.invalid_element_name = invalid_element_name
        self.tag_name = tag_name
        self.parent_name = parent_name

    def __str__(self):
        return 'Invalid name %s found in tag %s within element %s' % (repr(self.invalid_element_name),
                                                                      repr(self.tag_name),
                                                                      repr(self.parent_name))


def existing_file(filename):
    """
    Argparse type validator that ensures the path points to an existing file.

    :param filename: path to validate (str)
    :return: the same path, if valid (str)
    """
    if os.path.isfile(filename):
        return filename
    else:
        raise argparse.ArgumentTypeError('%s is not a valid input file!' % filename)


def valid_relation(relation):
    """
    Argparse type validator for `--relations` entries of the form `attribute:REL`.

    :param relation: Input string in the form 'attribute:relation_name' (str)
    :return: Tuple[str, str] of (attribute, relation_name)
    """
    parts = [part for part in relation.split(':') if len(part) > 0]
    if len(parts) == 2:
        return tuple(parts)
    else:
        raise argparse.ArgumentTypeError('%s must have the form attribute:relation' % relation)


def parse_args():
    """
    Parse CLI arguments.

    :return: argparse.Namespace
                Parsed arguments with fields:
                - xml_filename: str
                - dtd_filename: str
                - outputfile: str
                - annotate: bool
                - neo4j: bool
                - relations: Dict[str, str] (attribute -> relation)
    """
    parser = argparse.ArgumentParser(description='Parse the DBLP XML file and convert it to CSV')
    parser.add_argument('xml_filename', action='store', type=existing_file, help='The XML file that will be parsed',
                        metavar='xml_filename')
    parser.add_argument('dtd_filename', action='store', type=existing_file,
                        help='The DTD file used to parse the XML file', metavar='dtd_filename')
    parser.add_argument('outputfile', action='store', type=str, help='The output CSV file', metavar='outputfile')
    parser.add_argument('--annotate', action='store_true', required=False,
                        help='Write a separate annotated header with type information')
    parser.add_argument('--neo4j', action='store_true', required=False,
                        help='Headers become more Neo4J-specific and a neo4j-import shell script is generated for easy '
                             'importing. Implies --annotate.')
    parser.add_argument('--relations', action='store', required=False, type=valid_relation, nargs='+',
                        help='The element attributes that will be treated as elements, and to which a relation from '
                             'the parent element will be created. For example, in order to turn the author attribute '
                             'of the article element into an element with a relation, use "author:authored_by". The '
                             'part after the colon is used as the name of the relation.')
    parsed_args = parser.parse_args()

    # --neo4j implies --annotate
    if parsed_args.neo4j:
        if not parsed_args.annotate:
            parsed_args.annotate = True
            print('--neo4j implies --annotate!')

    # Normalize relations into a dict {attribute: relation}
    if parsed_args.relations:
        attr_rel = {attribute: relation for (attribute, relation) in parsed_args.relations}
        attributes = attr_rel.keys()

        # Ensure both attribute names and relation names are unique
        if len(attributes) == len(set(attr_rel.values())) == len(parsed_args.relations):
            parsed_args.relations = attr_rel
            print('Will create relations for attribute(s): %s' % (', '.join(sorted(attributes))))
        else:
            print('error: argument --relations: The element attributes and relation names must be unique.')
            exit(1)

    else:
        parsed_args.relations = dict()

    return parsed_args


def get_elements(dtd_file):
    """
    Read the DTD and return the set of element names (excluding the root 'dblp').

    :param dtd_file: Open file handle to the DTD (file-like)
    :return: Element names present in the DTD (Set[str])
    """
    dtd = etree.DTD(dtd_file)
    elements = set()

    for el in dtd.iterelements():
        if el.type == 'element':
            elements.add(el.name)
    elements.remove('dblp')

    return elements


def open_outputfiles(elements, element_attributes, output_filename, annotated=False):
    """
    Open a CSV writer per element that has discovered attributes.

    :param elements: Element types to write (e.g., 'article', 'inproceedings') (Set[str])
    :param element_attributes: Discovered attributes for each element (columns) (Dict[str, Set[str])
    :param output_filename: Base output file; actual files will be '<base>_<element><ext>' (str)
    :param annotated: If True, do not write the header row now (headers will be written in an annotated file later) (bool)
    :return: Mapping element -> CSV DictWriter (Dict[str, csv.DictWriter])
    """
    (path, ext) = os.path.splitext(output_filename)
    output_files = dict()

    for element in elements:
        fieldnames = element_attributes.get(element, None)
        if fieldnames is not None and len(fieldnames) > 0:
            fieldnames = sorted(list(fieldnames))
            fieldnames.insert(0, 'id')
            output_path = '%s_%s%s' % (path, element, ext)
            output_file = open(output_path, mode='w', encoding='UTF-8')
            output_writer = csv.DictWriter(output_file, fieldnames=fieldnames, delimiter=';',
                                           quoting=csv.QUOTE_MINIMAL, quotechar='"', doublequote=True,
                                           restval='', extrasaction='raise')
            if not annotated:
                output_writer.writeheader()

            output_files[element] = output_writer

    return output_files


def get_element_attributes(xml_file, elements):
    """
    Discover (and validate) per-element attributes by scanning the XML once.

    For each element:
    - Collects attribute keys on the element itself.
    - Collects child element names as columns.
    - For child elements with attributes, adds columns named 'child-attr'.
    - For multi-valued child tags, later rows merge with '|' separator.

    :param xml_file: Open file handle to the XML (file-like)
    :param elements: Element names to inspect (Set[str])
    :return: Mapping element -> discovered column names (Dict[str, Set[str])
    """
    context = etree.iterparse(xml_file, dtd_validation=True, events=('start', 'end'), attribute_defaults=True,
                              load_dtd=True)
    # turn it into an iterator
    context = iter(context)

    # get the root element
    event, root = next(context)
    data = dict()
    for element in elements:
        data[element] = set()
    current_tag = None

    for event, elem in context:
        if current_tag is None and event == 'start' and elem.tag in elements:
            current_tag = elem.tag
            keys = elem.keys()
            if len(keys) > 0:
                keys = set(keys)
                attributes = data[current_tag]
                attributes.update(keys)
        elif current_tag is not None and event == 'end':
            if elem.tag == current_tag:
                current_tag = None
            elif elem.tag is not None and elem.text is not None:
                if elem.tag == 'id':
                    raise InvalidElementName('id', elem.tag, current_tag)
                attributes = data[current_tag]
                attributes.add(elem.tag)
                keys = elem.keys()
                if len(keys) > 0:
                    for key in keys:
                        attributes.add('%s-%s' % (elem.tag, key))
            root.clear()

    # Prune empty, and protect preserved 'id'
    for element in elements:
        attributes = data[element]
        if len(attributes) == 0:
            data.pop(element)
        elif 'id' in attributes:
            raise InvalidElementName('id', element, 'root')

    return data


def parse_xml(xml_file, elements, output_files, relation_attributes, annotate=False):
    """
    Parse the XML and write element rows to CSV.

    :param xml_file: Open XML file handle (file-like)
    :param elements: Element types to parse/write (Set[str])
    :param output_files: Mapping element -> CSV DictWriter created by `open_outputfiles` (Dict[str, csv.DictWriter])
    :param relation_attributes: Attributes to extract as separate node sets (if any) (Set[str])
    :param annotate: If True, also collect: `array_elements` (columns that had multiple values), `element_types` (inferred types per element/column) (bool)
    :return:
        relations: For each relation attribute, map value -> set(ids of parent rows) (Dict[str, Dict[str, Set[int]]])
        unique_id: The next available unique id after the last written row (int)
        array_elements: For each element, the set of columns that were multi-valued (Dict[str, Set[str]])
        element_types: For each element/column, the set of raw types observed (e.g., {'integer', 'string'}) (Dict[str, Dict[str, Set[str]]])
    """
    context = etree.iterparse(xml_file, dtd_validation=True, events=('start', 'end'))
    # turn it into an iterator
    context = iter(context)

    # get the root element
    event, root = next(context)

    data = dict()
    relations = dict()
    current_tag = None
    multiple_valued_cells = set()
    unique_id = 0

    if annotate:
        array_elements = dict()
        element_types = dict()

    for event, elem in context:
        if current_tag is None and event == 'start' and elem.tag in elements:
            current_tag = elem.tag
            data.clear()
            multiple_valued_cells.clear()
            data.update(elem.attrib)
            if annotate:
                for (key, value) in elem.attrib.items():
                    set_type_information(element_types, current_tag, key, value)

        elif current_tag is not None and event == 'end':
            if elem.tag == current_tag:
                if len(data) > 0:
                    set_relation_values(relations, data, relation_attributes, unique_id)

                    # Join any multi-valued cells with '|'
                    for cell in multiple_valued_cells:
                        # mypy: data[cell] must exist and be a list here
                        data[cell] = '|'.join(data[cell])   # type: ignore[index]

                    data['id'] = unique_id
                    output_files[current_tag].writerow(data)    # type: ignore[arg-type]

                    if annotate and len(multiple_valued_cells) > 0:
                        element_cells = array_elements.get(current_tag)
                        if element_cells is None:
                            array_elements[current_tag] = multiple_valued_cells.copy()
                        else:
                            element_cells.update(multiple_valued_cells)

                    unique_id += 1

                current_tag = None

            elif elem.tag is not None and elem.text is not None:
                set_cell_value(data, elem.tag, elem.text, multiple_valued_cells)

                if annotate:
                    set_type_information(element_types, current_tag, elem.tag, elem.text)
                for (key, value) in elem.attrib.items():
                    column_name = '%s-%s' % (elem.tag, key)
                    set_cell_value(data, column_name, value, multiple_valued_cells)
                    if annotate:
                        set_type_information(element_types, current_tag, column_name, value)

            root.clear()

    if annotate:
        return relations, unique_id, array_elements, element_types
    else:
        return relations, unique_id


def set_relation_values(relations, data, relation_attributes, to_id):
    """
    Update relation index for configured attributes.

    For each `column_name` in `relation_attributes`, map its value(s) to the
    parent row id (`to_id`).

    :param relations: Accumulator for relations: attribute -> (value -> set(ids)) (Dict[str, Dict[str, Set[int]]])
    :param data: Current row's column data (Dict[str, Union[str, List[str]]])
    :param relation_attributes: Attribute column names to treat as separate node sets (Set[str])
    :param to_id: Parent row id to record as the END of the relation (int)
    :return: None
    """
    if len(relation_attributes) == 0:
        return

    for column_name, attributes in data.items():
        if column_name in relation_attributes:
            relation = relations.get(column_name, dict())
            if isinstance(attributes, list):
                for attribute in attributes:
                    rel_instance = relation.get(attribute, set())
                    rel_instance.add(to_id)
                    relation[attribute] = rel_instance
            else:
                rel_instance = relation.get(attributes, set())
                rel_instance.add(to_id)
                relation[attributes] = rel_instance
            relations[column_name] = relation


def set_cell_value(data, column_name, value, multiple_valued_cells):
    """
    Insert or append a cell value, tracking multi-valued columns.

    If a column is encountered multiple times for a row, it becomes a list and
    is later joined using '|'.

    :param data: Row data accumulator (Dict[str, Union[str, List[str]]])
    :param column_name: Column name (str)
    :param value: Value to insert/append (str)
    :param multiple_valued_cells: Set tracking which columns ended up with multiple values (Set[str])
    :return: None
    """
    entry = data.get(column_name)
    if entry is None:
        data[column_name] = value
    else:
        if isinstance(entry, list):
            entry.append(value)
        else:
            data[column_name] = [entry, value]
            multiple_valued_cells.add(column_name)


def set_type_information(element_types, current_tag, column_name, value):
    """
    Update observed raw types for (element, column).

    :param element_types: Mapping element -> (column -> observed raw types) (Dict[str, Dict[str, Set[str]]])
    :param current_tag: Current element type being parsed (str)
    :param column_name: Column being updated (str)
    :param value: Value to classify (integer/float/boolean/string/date/datetime/any) (str)
    :return: None
    """
    attribute_types = element_types.get(current_tag)
    if attribute_types is None:
        element_types[current_tag] = attribute_types = dict()
    types = attribute_types.get(column_name)
    if types is None:
        attribute_types[column_name] = types = set()
    types.add(get_type(value))


def get_type(string_value):
    """
    Infer a coarse scalar type for a string.

    Handles: integer, float, boolean, date (YYYY-MM-DD), datetime
    (YYYY-MM-DD HH:MM[:SS]), string, any (empty).

    :param string_value: Value to classify (str)
    :return: One of {'integer','float','boolean','date','datetime','string','any'} (str)
    """
    if string_value is None or len(string_value) == 0:
        return 'any'
    if str.isdigit(string_value):
        try:
            int(string_value)
            return 'integer'
        except ValueError:
            return 'string'
    if get_type.re_number.fullmatch(string_value) is not None:
        try:
            float(string_value)
            return 'float'
        except ValueError:
            return 'string'
    if get_type.re_date.fullmatch(string_value) is not None:
        try:
            date.fromisoformat(string_value)
            return 'date'
        except ValueError:
            return 'string'
    if get_type.re_datetime.fullmatch(string_value) is not None:
        try:
            datetime.fromisoformat(string_value)
            return 'datetime'
        except ValueError:
            return 'string'
    if string_value.lower() == 'true' or string_value.lower() == 'false':
        return 'boolean'
    return 'string'

# Precompiled regexes for `get_type`
get_type.re_number = re.compile(r'^\d+\.\d+$')
get_type.re_date = re.compile(r'^\d{4}-\d{2}-\d{2}$')
get_type.re_datetime = re.compile(r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}(?::\d{2})?$')


def write_annotated_header(array_elements, element_types, output_filename, neo4j_style=False):
    """
    Write annotated header files per element (one file per element).

    Each header row has 'column:Type' (or 'column:Type[]' for multi-valued).
    If `neo4j_style` is True, the first column is '<element>:ID' instead of 'id'.

    :param array_elements: Per-element multi-valued columns (Dict[str, Set[str]])
    :param element_types: Observed raw types per element/column (Dict[str, Dict[str, Set[str]]])
    :param output_filename: Base output file path used to construct per-element header file paths (str)
    :param neo4j_style: If True, use Neo4j ID type markers (bool)
    :return: None
    """
    (path, ext) = os.path.splitext(output_filename)
    for element, column_types in element_types.items():
        output_path = '%s_%s_header%s' % (path, element, ext)
        header = []
        array_columns = array_elements.get(element, set())
        columns = sorted(list(column_types.keys()))
        if neo4j_style:
            header.append('%s:ID' % element)
        else:
            columns.insert(0, 'id')
            column_types['id'] = {int}
        for column in columns:
            types = column_types[column]
            high_level_type = get_high_level_type(types)
            typename = translate_type(high_level_type, neo4j_style)
            if column in array_columns:
                header.append('%s:%s[]' % (column, typename))
            else:
                header.append('%s:%s' % (column, typename))
        with open(output_path, mode='w', encoding='UTF-8') as output_file:
            output_file.write(';'.join(header))


def translate_type(type_input, neo4j_style=False):
    """
    Translate to Neo4j type names where appropriate.

    :param type_input: High-level type ('integer','float','boolean','date','datetime','string') (str)
    :param neo4j_style: If True, map 'integer' -> 'int' (bool)
    :return: Possibly translated type name (str)
    """
    if neo4j_style and type_input == 'integer':
        return 'int'

    return type_input


def get_high_level_type(types: set) -> str:
    """
    Collapse a set of observed raw types to a single high-level type.

    Rules
    -----
    - Remove 'any' (empty) if others exist.
    - If only one type remains, return it.
    - If 'string' present, return 'string'.
    - If {'integer','float'} present, return 'float'.
    - If {'date','datetime'} present, return 'datetime'.
    - Otherwise return 'string'.

    :param types: Observed raw types (Set[str])
    :return: High-level collapsed type (str)
    """
    if len(types) == 0:
        raise Exception('Empty type set encountered', types)
    types.discard('any')
    if len(types) == 0:
        return 'string'
    elif len(types) == 1:
        (high_level_type,) = types
        return high_level_type
    else:
        if 'string' in types:
            return 'string'
        elif len(types) == 2:
            if 'float' in types and 'integer' in types:
                return 'float'
            elif 'date' in types and 'datetime' in types:
                return 'datetime'
    return 'string'


def generate_neo4j_import_command(elements, relations, relation_alias, output_filename):
    """
    Build a `neo4j-admin import` command string for the generated CSVs.

    :param elements: Elements for which node CSVs were generated (Set[str])
    :param relations: Attribute-node types for which CSVs were generated (from --relations) (Set[str])
    :param relation_alias: Mapping attribute -> relationship name (Dict[str, str])
    :param output_filename: Base output filename (str)
    :return: The assembled shell command (str)
    """
    (path, ext) = os.path.splitext(output_filename)
    command = 'neo4j-admin import --mode=csv --database=dblp.db --delimiter ";" --array-delimiter "|" ' \
              '--id-type INTEGER'
    for element in elements:
        command += ' --nodes:%s "%s_%s_header%s,%s_%s%s"' % (element, path, element, ext, path, element, ext)
    for relation in relations:
        command += ' --nodes:%s "%s_%s%s"' % (relation, path, relation, ext)
        predicate = relation_alias[relation]
        command += ' --relationships:%s "%s_%s_%s%s"' % (predicate, path, relation, predicate, ext)
    return command


def write_relation_files(output_filename, relations, relation_alias, unique_id):
    """
    Write node and relationship CSVs for configured attribute relations.

    For each attribute (e.g., 'author'):
    - Writes a node CSV with rows (synthetic id; value).
    - Writes a relation CSV with rows (:START_ID;:END_ID) linking parent rows
      to the attribute node.

    :param output_filename: Base output filename (str)
    :param relations: attribute -> (value -> set(parent_ids)) (Dict[str, Dict[str, Set[int]]])
    :param relation_alias: attribute -> relationship name (e.g., 'author' -> 'authored_by') (Dict[str, str])
    :param unique_id: Starting id for attribute nodes (will be incremented as nodes are written) (int)
    :return: None
    """
    (path, ext) = os.path.splitext(output_filename)
    for column_name, relation in relations.items():
        output_path_node = '%s_%s%s' % (path, column_name, ext)
        output_path_relation = '%s_%s_%s%s' % (path, column_name, relation_alias[column_name], ext)
        with open(output_path_relation, mode='w', encoding='UTF-8') as output_file_relation:
            output_file_relation.write(':START_ID;:END_ID\n')
            with open(output_path_node, mode='w', encoding='UTF-8') as output_file_node:
                node_output_writer = csv.writer(output_file_node, delimiter=';', quoting=csv.QUOTE_MINIMAL,
                                                quotechar='"', doublequote=True)
                output_file_node.write(':ID;%s:string\n' % column_name)
                for value, rel_instance in relation.items():
                    node_output_writer.writerow([unique_id, value])
                    for from_id in rel_instance:
                        output_file_relation.write('%d;%d\n' % (from_id, unique_id))
                    unique_id += 1


def main():
    """
    Entrypoint: parse args, discover schema, parse XML, write CSVs (and optional artifacts).

    Steps
    -----
    1) Read DTD and enumerate element types.
    2) First pass over XML to discover per-element attributes/columns.
    3) Open per-element CSV writers (and optionally skip direct headers if annotating).
    4) Second pass over XML to write rows and (optionally) collect type info.
    5) Optionally write relation CSVs.
    6) Optionally write annotated headers (and generate Neo4j import command/script).

    :return: None
    """
    args = parse_args()
    if args.xml_filename is not None and args.dtd_filename is not None and args.outputfile is not None:

        start_time = time.time()
        print('Start!')

        with open(args.dtd_filename, mode='rb') as dtd_file:
            print('Reading elements from DTD file...')
            elements = get_elements(dtd_file)

        with open(args.xml_filename, mode='rb') as xml_file:
            print('Finding unique attributes for all elements...')
            try:
                element_attributes = get_element_attributes(xml_file, elements)
            except InvalidElementName as e:
                element_attributes = None
                print(e)
                exit(1)

        print('Opening output files...')
        output_files = open_outputfiles(elements, element_attributes, args.outputfile, args.annotate)

        array_elements = None
        element_types = None
        with open(args.xml_filename, mode='rb') as xml_file:
            print('Parsing XML and writing to CSV files...')
            relation_attributes = set(args.relations.keys())
            if args.annotate:
                (relations, unique_id, array_elements, element_types) = parse_xml(xml_file, elements, output_files,
                                                                                  relation_attributes, annotate=True)
            else:
                relations, unique_id = parse_xml(xml_file, elements, output_files, relation_attributes)

        if args.relations and relations and unique_id >= 0:
            print('Writing relation files...')
            write_relation_files(args.outputfile, relations, args.relations, unique_id)

        if args.annotate and array_elements and element_types:
            print('Writing annotated headers...')
            write_annotated_header(array_elements, element_types, args.outputfile, args.neo4j)

            if args.neo4j:
                print('Generating neo4j-import command...')
                command = generate_neo4j_import_command(set(element_types.keys()), set(relations.keys()),
                                                        args.relations, args.outputfile)

                print('Writing neo4j-import command to shell script file...')
                with open('neo4j_import.sh', mode='w', encoding='UTF-8') as command_file:
                    command_file.write('#!/bin/bash\n')
                    command_file.write(command)

        end_time = time.time()

        print('Done after %f seconds' % (end_time - start_time))

    else:
        print('Invalid input arguments.')
        exit(1)


if __name__ == '__main__':
    main()