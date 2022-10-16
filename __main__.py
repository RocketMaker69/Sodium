from random import *
from enum import *
from time import *
from sys import *
import math



#####################################
# Errors
#####################################



running = True

def error(type: str, msg: str, pos: tuple | None):
	global running
	print(type + " " + str(pos or "") + ":\n  " + msg)
	running = False



#####################################
# Debug
#####################################



debug_settings = {
	"print_tokens": False,
	"print_ast":    False
}



#####################################
# Tokens
#####################################



class TokenType(Enum):
	EOF     = "EOF"
	NEWLINE = "NEWLINE"

	NUMBER  = "NUMBER"
	STRING  = "STRING"
	IDENT   = "IDENT"
	KEYWORD = "KEYWORD"
	SYMBOL  = "SYMBOL"

class Token:
	def __init__(self, type_: TokenType, pos: tuple, value=None) -> None:
		self.type = type_
		self.value = value
		self.pos = pos
	
	def __repr__(self) -> str:
		return f"[{self.type.value}: {self.value} at {self.pos}]" if self.value else f"[{self.type.value} at {self.pos}]"



#####################################
# Lexer
#####################################



class Lexer:
	keywords = [
		"let",
		"if",
		"while",
		"for",
		"function",
		"return",
		"class",
		"foreach",
		"in"
	]

	def __init__(self, input: str):
		self.input = input + "\n"
		self.current = ""
		self.index = -1
		self.col = 0
		self.ln = 1
		self.nextChar()

	def nextChar(self):
		self.index += 1
		self.col += 1
		if self.index >= len(self.input):
			self.current = '\0'
		else:
			self.current = self.input[self.index]

	def peek(self):
		if self.index + 1 >= len(self.input):
			return '\0'
		return self.input[self.index + 1]
		
	def skipWhitespace(self):
		while self.current in " \n\t\r":
			if self.current == "\n":
				self.ln += 1
				self.col = 0
			self.nextChar()
		
	def skipComments(self):
		if self.current == "#":
			while self.current != "\n":
				self.nextChar()
			self.nextChar()

	def getToken(self) -> Token:
		token = None

		self.skipWhitespace()
		self.skipComments()
		if self.current in "+-*/%()=<>!{}[]:":
			if self.peek() == "=" and self.current in "=!<>":
				ch = self.current
				self.nextChar()
				token = Token(TokenType.SYMBOL, (self.ln, self.col), ch + self.current)
			elif self.peek() == "*" and self.current in "*":
				ch = self.current
				self.nextChar()
				token = Token(TokenType.SYMBOL, (self.ln, self.col), ch + self.current)
			else:
				token = Token(TokenType.SYMBOL, (self.ln, self.col), self.current)
		elif self.current == "\"":
			self.nextChar()
			start = self.index
			while self.current != "\"":
				self.nextChar()
			string = self.input[start:self.index]
			token = Token(TokenType.STRING, (self.ln, self.col), string)
		elif self.current.isdigit():
			start = self.index
			while self.peek().isdigit():
				self.nextChar()
			if self.peek() == ".":
				self.nextChar()
				while self.peek().isdigit():
					self.nextChar()
			number = float(self.input[start:self.index + 1])
			token = Token(TokenType.NUMBER, (self.ln, self.col), number)
		elif self.current.isalpha() or self.current == "_":
			start = self.index
			while self.peek().isalnum():
				self.nextChar()
			ident = self.input[start:self.index + 1]
			token = Token(self.checkKeyword(ident), (self.ln, self.col), ident)
		elif self.current == "\0":
			token = Token(TokenType.EOF, (self.ln, self.col))
		else:
			error("SyntaxError", f"Undefined character: '{self.current}'.", (self.ln, self.col))
		
		self.nextChar()
		return token

	def checkKeyword(self, ident):
		if ident in self.keywords: 
			return TokenType.KEYWORD
		return TokenType.IDENT

	def lex(self) -> list[Token]:
		tokens = []

		while True:
			tok = self.getToken()
			if tok != None:
				tokens.append(tok)
				if tok.type == TokenType.EOF:
					break
			

		if debug_settings["print_tokens"]: print(tokens)
		return tokens



#####################################
# Nodes
#####################################



class Node:
	pass

class NumberNode(Node):
	def __init__(self, val) -> None:
		self.val = val

	def __repr__(self) -> str:
		return f"(Number: {self.val})"

class StringNode(Node):
	def __init__(self, val) -> None:
		self.val = val

	def __repr__(self) -> str:
		return f"(String: {self.val})"

class IndexNode(Node):
	def __init__(self, target, index, pos) -> None:
		self.pos    = pos
		self.target = target
		self.index  = index
	
	def __repr__(self) -> str:
		return f"(Get {self.target} at {self.index} at {self.pos})"

class BinaryOpNode(Node):
	def __init__(self, l, o, r) -> None:
		self.left  = l
		self.oper  = o
		self.right = r
	
	def __repr__(self) -> str:
		return f"({self.left} {self.oper} {self.right})"

class GetNode(Node):
	def __init__(self, var: str) -> None:
		self.var = var
	
	def __repr__(self) -> str:
		return f"(Get '{self.var}')"

class CallNode(Node):
	def __init__(self, pos: tuple, callee: Node, args: list[Node]) -> None:
		self.pos = pos
		self.callee = callee
		self.args = args
	
	def __repr__(self) -> str:
		return f"(Call {self.callee} at {self.pos} with {self.args})"

class ListNode(Node):
	def __init__(self, list_: list, pos: tuple) -> None:
		self.list = list_
		self.pos  = pos
	
	def __repr__(self) -> str:
		return f"(List {self.list} at {self.pos})"

#=======================================#

class ReturnNode(Node):
	def __init__(self, expr: Node) -> None:
		self.expr = expr
	
	def __repr__(self) -> str:
		return f"(Return {self.expr})"

class BlockNode(Node):
	def __init__(self, nodes: list[Node]) -> None:
		self.nodes = nodes

class LetNode(Node):
	def __init__(self, name: str, val: Node) -> None:
		self.name = name
		self.value = val
	
	def __repr__(self) -> str:
		return f"(Let {self.name} = {self.value})"

class AssignNode(Node):
	def __init__(self, name: str, val: Node) -> None:
		self.name = name
		self.value = val
	
	def __repr__(self) -> str:
		return f"({self.name} = {self.value})"

class IfNode(Node):
	def __init__(self, cond, stmt) -> None:
		self.cond = cond
		self.stmt = stmt
	
	def __repr__(self) -> str:
		return f"(If {self.cond}: {self.stmt})"

class ForeachNode(Node):
	def __init__(self, var, target, code, pos) -> None:
		self.var    = var
		self.target = target
		self.code   = code
		self.pos    = pos
	
	def __repr__(self) -> str:
		return f"(For each {self.var} in {self.target} at {self.pos}: {self.code})"

class WhileNode(Node):
	def __init__(self, cond, stmt) -> None:
		self.cond = cond
		self.stmt = stmt
	
	def __repr__(self) -> str:
		return f"(While {self.cond}: {self.stmt})"

class LambdaNode(Node):
	def __init__(self, args, body, pos) -> None:
		self.pos = pos
		self.args = args
		self.body = body
	
	def __repr__(self) -> str:
		return f"(Lambda ({self.args}) {{ {self.body} }})"
	
	def Call(self, args, i):
		if len(args) != len(self.args):
			error("ArgumentError", f"Function expected {len(self.args)} argument(s), got {len(args)} argument(s).", self.pos)
		i.current = Environment(i.current)
		for arg in self.args:
			i.current.Set(arg, i.visit(args[self.args.index(arg)]))
		i.visit(self.body)
		i.current = i.current.parent

class FunctionNode(Node):
	def __init__(self, name, args, body, pos) -> None:
		self.name = name
		self.pos = pos
		self.args = args
		self.body = body
	
	def __repr__(self) -> str:
		return f"(Function {self.name}({self.args}) {{ {self.body} }})"
	
	def Call(self, args, i):
		if len(args) != len(self.args):
			error("ArgumentError", f"Function expected {len(self.args)} argument(s), got {len(args)} argument(s).", self.pos)
		i.current = Environment(i.current)
		for arg in self.args:
			i.current.Set(arg, i.visit(args[self.args.index(arg)]))
		i.visit(self.body)
		i.current = i.current.parent

class ClassNode(Node):
	def __init__(self, nameToken: Token, fields={}, methods={}) -> None:
		self.name = nameToken
		self.fields = fields
		self.methods = methods
	
	def __repr__(self) -> str:
		return f"(Class {self.name.value}: fields: {self.fields}, methods: {self.methods})"
	
	def Get(self, name: Token):
		if name.value in self.methods:
			return self.methods[name.value]
		elif name.value in self.fields:
			return self.fields[name.value]
		else:
			error("AttributeError", f"Cannot find specified attribute '{name.value}'.", name.pos)

class ProgramNode(Node):
	def __init__(self, nodes: list[Node]):
		self.nodes = nodes

class ForNode(Node):
	def __init__(self, init, cond, post, code) -> None:
		self.init = init
		self.post = post
		self.cond = cond
		self.code = code
	
	def __repr__(self) -> str:
		return f"(For {self.init}; {self.cond}; {self.post} {{ {self.code} }})"



#####################################
# Parser
#####################################



class Parser:
	def __init__(self, tokens: list[Token]):
		self.tokens: list[Token] = tokens
		self.current: Token      = tokens[0]
		self.index: int          = 0

	def checkToken(self, kind: TokenType):
		return self.current.type == kind

	def match(self, kind: TokenType):
		if not self.checkToken(kind):
			error("InvalidSyntaxError", f"Expected {kind.name}, got {self.current.type.name}.", self.current.pos)
		self.nextToken()
		return None

	def nextToken(self):
		self.index += 1
		self.current = Token(TokenType.EOF, (0, 0)) if self.index >= len(self.tokens) else self.tokens[self.index]

	def peekToken(self):
		return Token(TokenType.EOF, (0, 0)) if self.index + 1 >= len(self.tokens) else self.tokens[self.index + 1]
	
	def matches(self, type, val):
		return self.current.type == type and self.current.value == val
	
	def lastToken(self):
		return self.tokens[self.index - 1]
	
	def program(self):
		prog = []
		while not self.checkToken(TokenType.EOF):
			prog.append(self.statement())
		if debug_settings["print_ast"]:
			print(prog)
		return ProgramNode(prog)
	
	def statement(self):
		if self.current.value == "let":
			self.nextToken()
			idt = self.current.value
			self.match(TokenType.IDENT)
			if not self.matches(TokenType.SYMBOL, '='):
				error("InvalidSyntaxError", "Expected '=', got " + str(self.current.value or "'' (EOF)") + ".", self.current.pos)
			self.nextToken()
			val = self.expr()
			return LetNode(idt, val)

		elif self.current.value == "if":
			self.nextToken()
			cond = self.assign()
			stmt = self.statement()
			return IfNode(cond, stmt)

		elif self.current.value == "while":
			self.nextToken()
			cond = self.assign()
			stmt = self.statement()
			return WhileNode(cond, stmt)

		elif self.current.value == "function":
			self.nextToken()
			self.match(TokenType.IDENT)
			name = self.lastToken()
			if not self.matches(TokenType.SYMBOL, '('):
				error("InvalidSyntaxError", "Expected '(', got " + str(self.current.value or "'' (EOF)") + ".", self.current.pos)
			self.nextToken()
			params = []
			while True:
				self.match(TokenType.IDENT)
				params.append(self.lastToken().value)
				if self.matches(TokenType.SYMBOL, ')') or self.checkToken(TokenType.EOF): break
			if not self.matches(TokenType.SYMBOL, ')'):
				error("InvalidSyntaxError", "Expected ')', got " + str(self.current.value or "'' (EOF)") + ".", self.current.pos)
			self.nextToken()
			body = self.statement()
			return FunctionNode(name.value, params, body, name.pos)

		elif self.current.value == "for":
			self.nextToken()
			init = self.statement()
			cond = self.assign()
			post = self.statement()
			code = self.statement()
			return ForNode(init, cond, post, code)
		
		elif self.current.value == "return":
			self.nextToken()
			expr = self.expr()
			return ReturnNode(expr)
		
		elif self.current.value == "class":
			self.nextToken()
			if not self.checkToken(TokenType.IDENT):
				error("InvalidSyntaxError", "Expected IDENT, got " + self.current.type.name + ".", self.current.pos)
			name = self.current
			self.nextToken()
			
			if not self.matches(TokenType.SYMBOL, "{"):
				error("InvalidSyntaxError", "Expected '{', got " + str(self.current.value or "'' (EOF)") + ".", self.current.pos)
			self.nextToken()

			# ...
			
			if not self.matches(TokenType.SYMBOL, "}"):
				error("InvalidSyntaxError", "Expected '}', got " + str(self.current.value or "'' (EOF)") + ".", self.current.pos)
			return ClassNode(name)
		
		elif self.current.value == "foreach":
			self.nextToken()
			var = self.current.value
			if not self.checkToken(TokenType.IDENT):
				error("InvalidSyntaxError", "Expected identifier.", self.current.pos)
			self.nextToken()
			if not self.matches(TokenType.KEYWORD, "in"):
				error("InvalidSyntaxError", "Expected 'in', got" + str(self.current.value or "'' (EOF)") + ".", self.current.pos)
			self.nextToken()
			pos = self.current.pos
			target = self.assign()
			code = self.statement()
			return ForeachNode(var, target, code, pos)


		elif self.current.value == "{":
			self.nextToken()
			nodes = []
			while self.current.value != "}":
				if self.checkToken(TokenType.EOF):
					error("InvalidSyntaxError", "Unexpected EOF; expected '}'.", self.current.pos)
					break
				nodes.append(self.statement())
			self.nextToken()
			return BlockNode(nodes)

		else:
			expr = self.assign()
			return expr
	
	def assign(self):
		target = self.expr()
		if self.matches(TokenType.SYMBOL, '='):
			self.nextToken()
			value = self.expr()
			if not isinstance(target, GetNode):
				error("InvalidSyntaxError", "Assignment target must be an identifier.", self.current.pos)
			target = AssignNode(target.var, value)
		return target

	def expr(self):
		l = self.comparison()
		while self.current.value in ("+", "-"):
			op = self.current
			self.nextToken()
			r = self.comparison()
			l = BinaryOpNode(l, op, r)
		return l

	def comparison(self):
		l = self.term()
		while self.current.value in ("==", "!=", "<=", ">=", ">", "<"):
			op = self.current
			self.nextToken()
			r = self.term()
			l = BinaryOpNode(l, op, r)
		return l
	
	def term(self):
		l = self.atom()
		while self.current.value in ("*", "/"):
			op = self.current
			self.nextToken()
			r = self.atom()
			l = BinaryOpNode(l, op, r)
		return l
	
	def atom(self):
		l = self.call()
		while self.current.value in ("**", "%"):
			op = self.current
			self.nextToken()
			r = self.call()
			l = BinaryOpNode(l, op, r)
		return l
	
	def call(self):
		l = self.factor()
		pos = self.current.pos
		while self.current.value == "(":
			self.nextToken()
			args = []
			while self.current.value != ")" and self.current.type != TokenType.EOF:
				args.append(self.assign())
			if not self.matches(TokenType.SYMBOL, ")"):
				error("InvalidSyntaxError", "Expected ')', got " + str(self.current.value or "'' (EOF)") + ".", self.current.pos)
			self.nextToken()
			l = CallNode(pos, l, args)
		return l
	
	def factor(self):
		if self.checkToken(TokenType.NUMBER):
			tok = self.current
			self.nextToken()
			return NumberNode(tok.value)
		elif self.checkToken(TokenType.STRING):
			tok = self.current
			self.nextToken()
			string = StringNode(tok.value)
			return string
		elif self.current.value == "function":
			pos = self.current.pos
			self.nextToken()
			if not self.matches(TokenType.SYMBOL, '('):
				error("InvalidSyntaxError", "Expected '(', got " + str(self.current.value or "'' (EOF)") + ".", self.current.pos)
			self.nextToken()
			params = []
			while True:
				if self.matches(TokenType.SYMBOL, ')') or self.checkToken(TokenType.EOF): break
				self.match(TokenType.IDENT)
				params.append(self.lastToken().value)
			if not self.matches(TokenType.SYMBOL, ')'):
				error("InvalidSyntaxError", "Expected ')', got " + str(self.current.value or "'' (EOF)") + ".", self.current.pos)
			self.nextToken()
			stmt = self.statement()
			return LambdaNode(params, stmt, pos)
		elif self.checkToken(TokenType.IDENT):
			tok = self.current
			self.nextToken()
			get = GetNode(tok)
			return get
		elif self.current.value == "(":
			self.nextToken()
			expr = self.assign()
			if not self.matches(TokenType.SYMBOL, ')'):
				error("InvalidSyntaxError", "Expected ')', got " + str(self.current.value or "'' (EOF)") + ".", self.current.pos)
			self.nextToken()
			return expr
		elif self.current.value == "[":
			pos = self.current.pos
			self.nextToken()
			l = []
			while not self.matches(TokenType.SYMBOL, "]") or self.checkToken(TokenType.EOF):
				l.append(self.assign())
			if not self.matches(TokenType.SYMBOL, ']'):
				error("InvalidSyntaxError", "Expected ']', got " + str(self.current.value or "'' (EOF)") + ".", self.current.pos)
			self.nextToken()
			list_ = ListNode(l, pos)
			return list_
		else:
			if not self.checkToken(TokenType.EOF):
				error("InvalidSyntaxError", "Unexpected token: " + repr(self.current) + ".", self.current.pos)
				self.nextToken()



##################################################################################
# RUNTIME:
##################################################################################

class Return(Exception):
	pass

#####################################
# Values
#####################################



class Value:
	def __init__(self) -> None:
		pass
	
	def add(self, other):
		return None

	def sub(self, other):
		return None
	
	def mul(self, other):
		return None

	def div(self, other):
		return None
	
	def pow(self, other):
		return None

	def mod(self, other):
		return None
	
	def eq(self, other):
		return None
	
	def neq(self, other):
		return None
	
	def gteq(self, other):
		return None
	
	def lteq(self, other):
		return None
	
	def gt(self, other):
		return None
	
	def lt(self, other):
		return None

	def __repr__(self) -> str:
		return ""



class Function(Value):
	def __init__(self, value: FunctionNode | LambdaNode) -> None:
		self.value = value
	
	def __repr__(self) -> str:
		return f"<Function {self.value.name}>" if isinstance(self.value, FunctionNode) else "<Lambda function>"



class List(Value):
	def __init__(self, value: list) -> None:
		self.value = value
	
	def Iter(self):
		return self.value
	
	def eq(self, other):
		return self.__class__(self.value == other.value)
	
	def neq(self, other):
		return self.__class__(self.value != other.value)

	def __repr__(self) -> str:
		return str(self.value)



class Boolean(Value):
	def __init__(self, value: bool) -> None:
		self.value = value
	
	def eq(self, other):
		return self.__class__(self.value == other.value)
	
	def neq(self, other):
		return self.__class__(self.value != other.value)
	
	def gteq(self, other):
		return self.__class__(self.value >= other.value)
	
	def lteq(self, other):
		return self.__class__(self.value <= other.value)
	
	def gt(self, other):
		return self.__class__(self.value > other.value)
	
	def lt(self, other):
		return self.__class__(self.value < other.value)

	def __repr__(self) -> str:
		return str(self.value)



class String(Value):
	def __init__(self, value: str) -> None:
		self.value = value
	
	def Iter(self):
		return self.value
	
	def add(self, other):
		return self.__class__(self.value + other.value)

	def mul(self, other):
		return self.__class__(self.value * other.value)
	
	def __repr__(self) -> str:
		return str(self.value)



class Number(Value):
	def __init__(self, value: float) -> None:
		self.value = value
	
	def add(self, other):
		return self.__class__(self.value + other.value)

	def sub(self, other):
		return self.__class__(self.value - other.value)
	
	def mul(self, other):
		return self.__class__(self.value * other.value)

	def div(self, other):
		if other.value != 0:
			return self.__class__(self.value / other.value)
		elif other.value == 0:
			return math.inf
		elif self.value == 0 and other.value == 0:
			return math.nan
	
	def pow(self, other):
		return self.__class__(self.value ** other.value)

	def mod(self, other):
		return self.__class__(self.value % other.value)
	
	def eq(self, other):
		return Boolean(self.value == other.value)
	
	def neq(self, other):
		return Boolean(self.value != other.value)
	
	def gteq(self, other):
		return Boolean(self.value >= other.value)
	
	def lteq(self, other):
		return Boolean(self.value <= other.value)
	
	def gt(self, other):
		return Boolean(self.value > other.value)
	
	def lt(self, other):
		return Boolean(self.value < other.value)
	
	def __repr__(self) -> str:
		return str(self.value)

class Class(Value):
	def __init__(self, value: ClassNode) -> None:
		self.value = value
	
	def __repr__(self) -> str:
		return f"<Class '{self.value.name}'>"



#####################################
# Environment
#####################################



class Environment:
	def __init__(self, parent = None) -> None:
		self.values = {}
		self.parent = parent
	
	def Get(self, __name: Token):
		try:
			return self.values[__name.value]
		except KeyError:
			if self.parent:
				return self.parent.Get(__name)
			else:
				error("SymbolError", f"Cannot find specified symbol '{__name.value}'.", __name.pos)
	
	def Assign(self, __name: Token, __value) -> None:
		if __name.value in self.values:
			self.values[__name.value] = __value
		elif self.parent:
			self.parent.Assign(__name, __value)
		else:
			error("SymbolError", f"Cannot find specified symbol '{__name.value}'.", __name.pos)
	
	def Set(self, __name: str, __value) -> None:
		self.values[__name] = __value



#####################################
# Interpreter
#####################################



def toNode(val):
	if isinstance(val, String | str):
		return StringNode(val)
	elif isinstance(val, Number | float):
		return NumberNode(val)
	elif isinstance(val, List | list):
		return ListNode(val)
	return val

class Interpreter:
	def __init__(self):
		self.globals = Environment()

		self.globals.Set("pi",  Number(math.pi))
		self.globals.Set("e",   Number(math.e))
		self.globals.Set("Phi", Number((1 + math.sqrt(5)) / 2))
		self.globals.Set("phi", Number((1 - math.sqrt(5)) / 2))
		self.globals.Set("tau", Number(math.tau))
		
		self.globals.Set("Inf", Number(math.inf))
		self.globals.Set("NaN", Number(math.nan))
		
		self.current = Environment(self.globals)

	def visit(self, node):
		global running
		if running:
			visit_fn = getattr(self, f"visit{type(node).__name__}", self.visitGeneric)
			val = visit_fn(node)
			return val
	
	def visitGeneric(self, node):
		error("InternalError", f"Cannot find node method visit{type(node).__name__}(...).", None)
	
	def visitProgramNode(self, node):
		outs = []
		for n in node.nodes:
			outs.append(self.visit(n))
		return outs
	
	def visitBlockNode(self, node):
		self.current = Environment(self.current)
		for node_ in node.nodes:
			self.visit(node_)
		self.current = self.current.parent
		return None

	def visitNumberNode(self, node: NumberNode):
		return Number(node.val)
	
	def visitStringNode(self, node: StringNode):
		return String(node.val)
	
	def visitBinaryOpNode(self, node: BinaryOpNode):
		l: Value = self.visit(node.left)
		r: Value = self.visit(node.right)

		match node.oper.value:
			case "+":  return l.add(r)
			case "-":  return l.sub(r)
			case "*":  return l.mul(r)
			case "/":  return l.div(r)
			case "**": return l.pow(r)
			case "%":  return l.mod(r)
			case "==": return l.eq(r)
			case "!=": return l.neq(r)
			case ">=": return l.gteq(r)
			case "<=": return l.lteq(r)
			case ">":  return l.gt(r)
			case "<":  return l.lt(r)
			case _:
				error("InternalError", f"Undefined operator '{node.oper.value}'.", node.oper.pos)
	
	def visitLetNode(self, node: LetNode):
		self.current.Set(node.name, self.visit(node.value))
	
	def visitForeachNode(self, node: ForeachNode):
		iter_ = self.visit(node.target)
		if not hasattr(iter_, "Iter"):
			error("TypeError", "Expected iterable.", node.pos)
		else:
			iter = iter_.Iter()
			for i in iter:
				self.visit(LetNode(node.var, toNode(i)))
				self.visit(node.code)

	
	def visitLambdaNode(self, node: LambdaNode):
		return Function(node)
	
	def visitIndexNode(self, node: IndexNode):
		target = self.visit(node.target)
		if not isinstance(target, String | List):
			error("TypeError", "Object must be indexable (string, variable holding a list or a list).", node.pos)
		else:
			idx = self.visit(node.index).value
			if int(idx) != idx:
				error("TypeError", "Index must be an integer.", node.pos)
			elif int(idx) >= len(target.value):
				error("IndexError", "Index out of range.", node.pos)
			else:
				return target.value[int(idx)]
	
	def visitListNode(self, node: ListNode):
		_list = []
		for i in node.list:
			_list.append(self.visit(i))
		return List(_list)

	def visitAssignNode(self, node: AssignNode):
		self.current.Assign(node.name, self.visit(node.value))
	
	def visitGetNode(self, node: GetNode):
		val = self.current.Get(node.var)
		if val: return val
	
	def visitFunctionNode(self, node: FunctionNode):
		self.current.Set(node.name, Function(node))
	
	def visitCallNode(self, node: CallNode):
		pos = node.pos
		callee = self.visit(node.callee)
		if callee != None:
			if callee:
				try:
					if isinstance(callee, Value):
						callee.value.Call(node.args, self)
					else:
						callee.Call(node.args, self, pos)
				except AttributeError:
					error("SymbolError", f"Cannot call non-callable object {callee}.", None)
				except Return as r:
					return self.visit(r.args[0])
	
	def visitReturnNode(self, node: ReturnNode):
		raise Return(node.expr)
	
	def visitClassNode(self, node: ClassNode):
		self.current.Set(node.name, Class(node))
	
	def visitIfNode(self, node: IfNode):
		if self.visit(node.cond).value:
			self.visit(node.stmt)

	def visitWhileNode(self, node: WhileNode):
		while self.visit(node.cond).value:
			self.visit(node.stmt)
	
	def visitForNode(self, node: ForNode):
		self.visit(node.init)
		while self.visit(node.cond).value:
			self.visit(node.code)
			self.visit(node.post)

interp = Interpreter()



#####################################
# Built-ins
#####################################



def define(name: str):
	def _define(value):
		interp.globals.Set(name, value)
	return _define

# Println(s...)
@define("Println")
class _Println:
	def Call(args, i, p):
		for arg in args:
			print(i.visit(arg), end="")
		print("")

# Input(s?)
@define("Input")
class _Input:
	def Call(args, i, p):
		value = input(args[0] if len(args) >= 1 else "")
		raise Return(String(value))

# Run(f)
@define("Run")
class _Run:
	def Call(args, i, p):
		if len(args) != 1:
			error("TypeError", f"Run(...) expected 1 argument, got {len(args)} argument(s).", p)
			return
		file = args[0]
		if not isinstance(file, StringNode):
			error("TypeError", f"Run(...) expected string.", p)
		code = open(i.visit(file).value, "r").read()

		lexer = Lexer(code)
		parser = Parser(lexer.lex())
		i.visit(parser.program())

# Wait(t)
@define("Wait")
class _Wait:
	def Call(args, i, p):
		if len(args) != 1:
			error("TypeError", f"Run(...) expected 1 argument, got {len(args)} argument(s).", p)
			return
		t = args[0]
		if not isinstance(t, NumberNode):
			error("TypeError", f"Run(...) expected number.", p)
		sleep(i.visit(t).value)

# Random(end)
@define("Random")
class _Random:
	def Call(args, i, p):
		if len(args) != 1:
			error("TypeError", f"Random(...) expected 1 argument, got {len(args)} argument(s).", p)
		end = args[0]
		if not isinstance(end, NumberNode):
			error("TypeError", f"Random(...) arg 0 expected to be a number.", p)
		val = random() * i.visit(end).value
		raise Return(Number(val))

# Append(list, item)
@define("Append")
class _Append:
	def Call(args, i, p):
		if len(args) != 2:
			error("TypeError", f"Append(...) expected 2 arguments, got {len(args)} argument(s).", p)
		list = i.visit(args[0])
		if not isinstance(list, List):
			error("TypeError", f"Append(...) arg 0 expected to be a list.", p)
		else:
			item = args[1]
			list.value.append(i.visit(item))

# Get(list, index)
@define("Get")
class _Get:
	def Call(args, i, p):
		if len(args) != 2:
			error("TypeError", f"Get(...) expected 2 arguments, got {len(args)} argument(s).", p)
		list = i.visit(args[0])
		if not isinstance(list, List):
			error("TypeError", f"Get(...) arg 0 expected to be a list.", p)
		else:
			idx = i.visit(args[1]).value
			if idx < len(list.value):
				value = NumberNode(list.value[int(idx)])
				raise Return(value)
			else:
				error("TypeError", "Index out of range.", p)

# Exit()
@define("Exit")
class _Exit:
	def Call(args, i, p):
		exit(0)



##################################################################################
# Main Code
##################################################################################



def repl():
	global running
	global interp
	if len(argv) == 2:
		lexer  = Lexer(open(argv[1], "r").read())
		parser = Parser(lexer.lex())
		ast = parser.program()
		if running:
			interp.visit(ast)

	else:
		print("Sodium 1.1 (Python 3.10.7 64-bit)\nCopyright 2022 RocketMaker69\n")
		while True:
			input_ = input(">>> ")
			lexer  = Lexer(input_)
			parser = Parser(lexer.lex())
			ast = parser.program()
			if running:
				values = interp.visit(ast)
				for val in values:
					if val != None:
						print(val)
			running = True

repl()